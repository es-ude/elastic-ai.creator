from abc import abstractmethod
from collections.abc import Iterable
from typing import override

import torch
import torch.nn
from elasticai.creator_plugins.lutron_filter.precompute.int_encoding_lutron_convolution import (
    LutronBitsToTwoComplementsWrapper,
)
from elasticai.creator_plugins.lutron_filter.precompute.lutron_filter import (
    LutronConv,
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
from elasticai.creator.ir import compose_rules

from ._ir import DataGraph, Node, NodeConstraint, Registry, sequential_with_interface
from ._precomputation import (
    FilterParameters,
    PrecomputationStrategy,
    make_precompute_rule,
)
from ._remove_redundant_layers import remove_redundant_layers
from ._shape_inference import InferMaxPool1dInChannelsRule


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
        return self.constraint_fn

    @abstractmethod
    def constraint_fn(self, pattern_node: Node, graph_node: Node, /) -> bool: ...

    @abstractmethod
    def build_lutron_module(self, filter_params: FilterParameters) -> LutronModule: ...

    @override
    def get_io_pairs(self) -> Iterable[Iterable[tuple[str, str]]]:

        filter_params = self.get_filter_parameters()
        module = self.build_lutron_module(filter_params)
        module.eval()
        inputs, outputs = module.generate_io_tensors()

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
    def constraint_fn(self, pattern_node: Node, graph_node: Node, /) -> bool:
        if pattern_node.type == "interface":
            return graph_node.type in ("binarize", "flatten", "sigmoid")
        return pattern_node.type == graph_node.type

    def build_lutron_module(self, filter_params: FilterParameters) -> LutronModule:
        graph = self.get_impl("linear")
        linear = torch.nn.Linear(
            in_features=filter_params.in_channels,
            out_features=filter_params.out_channels,
            bias=graph.attributes["bias"],
        )
        linear.weight.data = torch.tensor(
            graph.attributes["parameters"]["weight"], dtype=torch.float32
        )
        if linear.bias is not None:
            linear.bias.data = torch.tensor(
                graph.attributes["parameters"]["bias"], dtype=torch.float32
            )

        class RemoveKernelDim(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x.view(-1, filter_params.in_channels)

        class AddKernelDim(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x.view(-1, 1, filter_params.out_channels)

        lutron_linear = LutronLinear(
            wrapped=torch.nn.Sequential(RemoveKernelDim(), linear, AddKernelDim()),
            filter_parameters=filter_params,
        )
        lutron_linear.eval()
        return lutron_linear

    def get_filter_parameters(self) -> FilterParameters:
        g = self.get_impl("linear")
        return FilterParameters(
            in_channels=g.attributes["in_features"],
            kernel_size=1,
            out_channels=g.attributes["out_features"],
        )


precompute_linear = make_precompute_rule(PrecomputeLinear())


class _PrecomputeMaxPool(_BasePrecompute):
    def __init__(self):
        super().__init__(graph=sequential_with_interface("maxpool1d"))

    @override
    def constraint_fn(self, pattern_node: Node, graph_node: Node, /) -> bool:
        match pattern_node.name:
            case "start":
                if graph_node.type in ("conv1d", "binarize", "filter"):
                    return True
            case "end":
                return True
        return pattern_node.type == graph_node.type

    @override
    def get_filter_parameters(self) -> FilterParameters:
        maxpool_impl = self.get_impl("maxpool1d")
        attrs = maxpool_impl.attributes
        network, _ = self._module
        mp_node = network.nodes["maxpool1d"]
        in_channels = mp_node.attributes.get_int("in_channels")
        out_channels = in_channels
        return FilterParameters(
            kernel_size=attrs["kernel_size"],
            in_channels=in_channels,
            out_channels=out_channels,
            groups=in_channels,
            stride=attrs["stride"],
        )

    @override
    def build_lutron_module(self, filter_params: FilterParameters) -> LutronModule:
        return LutronMaxPool(
            torch.nn.MaxPool1d(
                kernel_size=filter_params.kernel_size,
                stride=filter_params.stride,
            ),
            filter_parameters=filter_params,
        )


precompute_maxpool = make_precompute_rule(_PrecomputeMaxPool())


def _get_conv_filter_params(precomp_strat: _BasePrecompute) -> FilterParameters:
    conv_impl = precomp_strat.get_impl("conv1d")
    attributes = conv_impl.attributes
    params = {}
    for k in ("kernel_size", "in_channels", "out_channels", "groups", "stride"):
        params[k] = attributes[k]

    p = FilterParameters(**params)
    p.in_channels = attributes.get_int("input_bitwidth", 1) * p.in_channels
    p.out_channels = attributes.get_int("output_bitwidth", 1) * p.out_channels
    return p


def _build_torch_conv1d(
    impl: DataGraph, filter_params: FilterParameters
) -> torch.nn.Module:
    # TODO: this can be drastically simplified now that parameters are
    # stored in their own sub-attribute
    parameters = impl.attributes.get_mapping("parameters")
    attrs = impl.attributes
    constr_args = {}
    for k in ("kernel_size", "groups"):
        constr_args[k] = filter_params.as_dict()[k]  # type: ignore
    input_bitwidth = attrs.get_int("input_bitwidth", 1)
    output_bitwidth = attrs.get_int("output_bitwidth", 1)
    constr_args["in_channels"] = filter_params.in_channels // input_bitwidth
    constr_args["out_channels"] = filter_params.out_channels // output_bitwidth
    if "bias" in impl.attributes:
        _bias = impl.attributes["bias"]
        if isinstance(_bias, bool):
            constr_args["bias"] = _bias
        elif _bias is not None:
            constr_args["bias"] = True
    torch_layer = torch.nn.Conv1d(**constr_args)
    weight = parameters["weight"]
    weight = torch.tensor(weight)
    torch_layer.weight.data = weight
    if (
        "bias" in parameters
        and hasattr(torch_layer, "bias")
        and torch_layer.bias is not None
        and not isinstance(parameters["bias"], bool)
    ):
        torch_layer.bias.data = torch.tensor(parameters["bias"])
    num_input_bits = impl.attributes["input_bitwidth"]
    if num_input_bits > 1:
        torch_layer = LutronBitsToTwoComplementsWrapper(num_input_bits, torch_layer)
    return torch_layer


def _conv_constraint(pattern_node: Node, graph_node: Node) -> bool:
    match pattern_node.name:
        case "start":
            return graph_node.type in ("binarize", "maxpool1d", "filter", "input")
        case "end":
            return graph_node.type in ("binarize", "sigmoid", "flatten")
        case _:
            return graph_node.type == pattern_node.type


class _PrecomputeConv1dVanilla(_BasePrecompute):
    def __init__(self):
        super().__init__(graph=sequential_with_interface("conv1d"))

    @override
    def constraint_fn(self, pattern_node: Node, graph_node: Node) -> bool:
        return _conv_constraint(pattern_node, graph_node)

    @override
    def get_filter_parameters(self) -> FilterParameters:
        return _get_conv_filter_params(self)

    @override
    def build_lutron_module(self, filter_params: FilterParameters) -> LutronModule:

        ir_layer = self.get_impl("conv1d")
        return LutronConv(_build_torch_conv1d(ir_layer, filter_params), filter_params)


class _PrecomputeConv1dBNorm(_BasePrecompute):
    def __init__(self):
        super().__init__(graph=sequential_with_interface("conv1d", "batchnorm1d"))

    @override
    def constraint_fn(self, pattern_node: Node, graph_node: Node) -> bool:
        return _conv_constraint(pattern_node, graph_node)

    @override
    def get_filter_parameters(self) -> FilterParameters:
        return _get_conv_filter_params(self)

    @override
    def build_lutron_module(self, filter_params: FilterParameters) -> LutronModule:
        def build_bnorm() -> torch.nn.Module:
            attr = self.get_impl("batchnorm1d").attributes
            bnorm = torch.nn.BatchNorm1d(
                num_features=attr.get_int("num_features"),
                affine=attr.get_bool("affine", False),
            )
            p = attr.get_mapping("parameters")
            bnorm.running_mean = torch.tensor(attr["running_mean"])
            bnorm.running_var = torch.tensor(attr["running_var"])
            if bnorm.affine:
                bnorm.weight.data = torch.tensor(p["weight"])
                bnorm.bias.data = torch.tensor(p["bias"])
            return bnorm

        ir_conv = self.get_impl("conv1d")
        torch_conv = _build_torch_conv1d(ir_conv, filter_params)
        torch_bnorm = build_bnorm()
        return LutronConv(torch.nn.Sequential(torch_conv, torch_bnorm), filter_params)


precompute = compose_rules(
    InferMaxPool1dInChannelsRule(),
    precompute_linear,
    precompute_maxpool,
    make_precompute_rule(_PrecomputeConv1dVanilla()),
    make_precompute_rule(_PrecomputeConv1dBNorm()),
    make_precompute_rule(_PrecomputeConv1dVanilla()),
    make_precompute_rule(_PrecomputeConv1dBNorm()),
    remove_redundant_layers,
    remove_redundant_layers,
)
