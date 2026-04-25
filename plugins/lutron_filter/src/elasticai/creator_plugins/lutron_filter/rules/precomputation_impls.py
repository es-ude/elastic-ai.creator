from abc import abstractmethod
from collections.abc import Iterable
from typing import override

import torch
import torch.nn
from elasticai.creator_plugins.lutron_filter.precompute.lutron_filter import (
    LutronLinear,
    LutronMaxPool,
)
from elasticai.creator_plugins.lutron_filter.precompute.lutron_module_protocol import (
    LutronModule,
)
from elasticai.creator_plugins.lutron_filter.tensor_conversion import (
    torch1d_input_tensor_to_grouped_strings,
)

import elasticai.creator.ir.datagraph_rewriting as _rew
from elasticai.creator.ir2vhdl import Shape

from ._ir import DataGraph, Node, NodeConstraint, Registry, sequential_with_interface
from .precomputation import (
    FilterParameters,
    PrecomputationStrategy,
    make_precompute_rule,
)


class _BasePrecompute(PrecomputationStrategy):
    def __init__(self, graph: _rew.DataGraph):
        self._g = graph

    @property
    @override
    def pattern_graph(self):
        return self._g

    @property
    def _match(self) -> DataGraph:
        return self._module[0]

    @property
    def _registry(self) -> Registry[DataGraph]:
        return self._module[1]

    @override
    def constraint(self, _: Registry[DataGraph]) -> NodeConstraint:
        return self._constraint_fn

    @abstractmethod
    def _constraint_fn(self, pattern_node: Node, graph_node: Node, /) -> bool: ...

    @abstractmethod
    def _build_lutron_module(self, filter_params: FilterParameters) -> LutronModule: ...

    @abstractmethod
    def _extract_filter_parameters(self) -> FilterParameters: ...

    @override
    def get_io_pairs(self) -> Iterable[Iterable[tuple[str, str]]]:
        graph, registry = self._module

        filter_params = self._extract_filter_parameters()
        inputs, outputs = self._build_lutron_module(filter_params).generate_io_tensors()

        def to_bits(x):
            x = torch.sign(x)
            x = ((x + 1) / 2).to(dtype=torch.int)
            return x

        inputs = to_bits(inputs)
        outputs = to_bits(outputs)
        inputs = torch1d_input_tensor_to_grouped_strings(
            inputs, groups=filter_params.groups
        )
        outputs = torch1d_input_tensor_to_grouped_strings(outputs, filter_params.groups)
        for input_group, output_group in zip(inputs, outputs):
            yield zip(input_group, output_group)


class PrecomputeLinear(_BasePrecompute):
    def __init__(self):
        super().__init__(sequential_with_interface("linear"))

    @override
    def _constraint_fn(self, pattern_node: Node, graph_node: Node, /) -> bool:
        if pattern_node.type == "interface":
            return graph_node.type in ("binarize",)
        return pattern_node.type == graph_node.type

    def _build_lutron_module(self, filter_params: FilterParameters) -> LutronModule:
        graph, _ = self._module
        linear = torch.nn.Linear(
            in_features=filter_params.in_channels,
            out_features=filter_params.out_channels,
            bias=graph.attributes["bias"],
        )
        linear.weight.data = torch.tensor(
            graph.attributes["parameters"]["weight"], dtype=torch.float32
        )
        linear.bias.data = torch.tensor(
            graph.attributes["parameters"]["bias"], dtype=torch.float32
        )

        class RemoveKernelDim(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x.view(-1, filter_params.in_channels)

        class AddKernelDim(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x.view(-1, filter_params.out_channels, 1)

        lutron_linear = LutronLinear(
            wrapped=torch.nn.Sequential(RemoveKernelDim(), linear, AddKernelDim()),
            filter_parameters=filter_params,
        )
        lutron_linear.eval()
        return lutron_linear

    def _extract_filter_params(self) -> FilterParameters:
        g = self._get_impl("linear")
        return FilterParameters(
            kernel_size=1,
            in_channels=g.attributes["in_features"],
            out_channels=g.attributes["out_features"],
        )


precompute_linear = make_precompute_rule(PrecomputeLinear())


class _PrecomputeMaxPool(_BasePrecompute):
    def __init__(self):
        super().__init__(graph=sequential_with_interface("maxpool1d"))

    @override
    def _constraint_fn(self, pattern_node: Node, graph_node: Node, /) -> bool:
        match pattern_node.name:
            case "start":
                if graph_node.type in ("conv1d", "binarize", "filter"):
                    return True
            case "end":
                return True
        return pattern_node.type == graph_node.type

    @override
    def _extract_filter_parameters(self) -> FilterParameters:
        maxpool_impl = self._get_impl("maxpool1d")
        attrs = maxpool_impl.attributes
        network, _ = self._module
        mp_node = network.nodes["maxpool1d"]
        in_shape = Shape(mp_node.attributes["input_size"])
        out_shape = Shape(mp_node.attributes["output_size"])
        assert in_shape.depth == out_shape.depth
        return FilterParameters(
            kernel_size=attrs["kernel_size"],
            in_channels=in_shape.depth,
            out_channels=out_shape.depth,
            groups=in_shape.depth,
            stride=attrs["stride"],
        )

    @override
    def _build_lutron_module(self, filter_params: FilterParameters) -> LutronModule:
        # MaxPool doesn't need a lutron module, return a dummy implementation
        graph, _ = self._module
        return LutronMaxPool(
            torch.nn.MaxPool1d(
                kernel_size=filter_params.kernel_size,
                stride=filter_params.stride,
            ),
            filter_parameters=filter_params,
        )
