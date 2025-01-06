import re
from typing import Callable, Iterator, cast

import torch.fx
from torch.nn import BatchNorm1d, Conv1d, MaxPool1d, Module

from elasticai.creator.ir import Edge, RequiredField
from elasticai.creator.ir import Node as _Node
from elasticai.creator.ir.helpers import Shape, ShapeTuple
from elasticai.creator.ir2vhdl import Implementation as _Implementation

from .nn.lutron.binarize import Binarize
from .nn.lutron.truth_table_generation import (
    convert_from_numbers_to_binary_logic,
)
from .torch import LutronBlockMatcher, LutronModule
from .torch.default_lutron_block_matcher import (
    LutronBlockMatcher as _DefaultLutronBlockMatcher,
)
from .torch.lutron_block_autogen import autogenerate_lutron_blocks
from .torch.lutron_modules import LutronFilter
from .torch.tensor_conversion import (
    torch1d_input_tensor_to_grouped_strings,
)
from .torch.transformation_utils import LeafTracer
from .torch.transformations import (
    remove_identities,
    reorder_conv_blocks,
)


def _shape_to_tuple(s: Shape) -> ShapeTuple:
    return s.to_tuple()


class Node(_Node):
    implementation: str
    input_shape: RequiredField[ShapeTuple, Shape] = RequiredField(
        _shape_to_tuple, Shape.from_tuple
    )
    output_shape: RequiredField[ShapeTuple, Shape] = RequiredField(
        _shape_to_tuple, Shape.from_tuple
    )


def _new_node(
    name: str,
    type: str,
    implementation: str,
    input_shape: Shape,
    output_shape: Shape,
    attributes: None | dict = None,
) -> Node:
    if attributes is None:
        attributes = {}
    n = Node(
        attributes
        | dict(
            name=name,
            type=type,
            implementation=implementation,
        )
    )
    n.input_shape = input_shape
    n.output_shape = output_shape

    return n


class Implementation(_Implementation[Node, Edge]):
    def __init__(self, name: str, type: str):
        super().__init__(name, type, {})

    @staticmethod
    def _new_node(
        name: str,
        type: str,
        implementation: str,
        input_shape: Shape,
        output_shape: Shape,
        attributes: None | dict = None,
    ) -> Node:
        if attributes is None:
            attributes = {}
        n = _new_node(
            name=name,
            type=type,
            input_shape=input_shape,
            output_shape=output_shape,
            implementation=implementation,
            attributes=attributes,
        )

        return n

    def input(self, size: int, channels: int):
        self.add_node(
            self._new_node(
                name="input",
                input_shape=Shape(channels, size),
                type="input",
                implementation="",
                output_shape=Shape(channels, size),
            )
        )

    def output(self, size: int, channels: int):
        self.add_node(
            self._new_node(
                name="output",
                input_shape=Shape(channels, size),
                type="output",
                implementation="",
                output_shape=Shape(channels, size),
            )
        )

    def lutron_filter(self, name, module: LutronFilter, implementation: str) -> None:
        self.add_node(
            self._new_node(
                name=name,
                type="grouped_filter",
                implementation=implementation,
                input_shape=Shape(
                    module.filter_parameters.in_channels,
                    module.filter_parameters.input_size,
                ),
                output_shape=Shape(
                    module.filter_parameters.out_channels,
                    module.filter_parameters.output_size,
                ),
                attributes={"params": module.filter_parameters.as_dict()},
            )
        )


class LutronFilterImplementation(_Implementation):
    def __init__(
        self, name: str, lutrons: dict[str, dict[tuple[int, ...], tuple[int, ...]]]
    ):
        super().__init__(name, type="grouped_filter", attributes=dict(lutrons=lutrons))


def create_new_lutron_cell_names(registry) -> Iterator[str]:
    def yield_lutron_suffixes(registry):
        yield 0
        for name in registry:
            match = re.match(r"lutron_(\d+)", name)
            if match:
                suffix = int(match.group(1))
                yield suffix

    max_suffix = max(yield_lutron_suffixes(registry))
    new_suffix = max_suffix + 1
    while True:
        yield f"lutron_{new_suffix}"
        new_suffix += 1


def add_lutron_filter_impl(
    name: str, module: Module, registry: dict[str, _Implementation[Node, Edge] | None]
) -> Iterator[_Implementation[Node, Edge]]:
    params = module.filter_parameters

    def generate_lutron_values():
        inputs, outputs = tuple(
            map(convert_from_numbers_to_binary_logic, module.generate_io_tensors())
        )
        inputs = torch1d_input_tensor_to_grouped_strings(inputs, groups=params.groups)
        outputs = torch1d_input_tensor_to_grouped_strings(outputs, groups=params.groups)
        for group_inputs, group_outputs in zip(inputs, outputs):
            yield tuple(zip(group_inputs, group_outputs))

    luts = []

    def register_name(name: str) -> None:
        registry[name] = None

    impl = _Implementation(
        name,
        "grouped_filter",
        attributes=dict(params=params.as_dict(), kernel_per_group=luts),
    )
    for _name, value in zip(
        create_new_lutron_cell_names(registry), generate_lutron_values()
    ):
        register_name(_name)
        input_size = len(value[0][0])
        output_size = len(value[0][1])
        yield _Implementation(
            _name,
            "lutron",
            attributes={
                "truth_table": value,
                "input_size": input_size,
                "output_size": output_size,
            },
        )
        luts.append(_name)
        impl.add_node(
            Node(
                dict(
                    name=_name,
                    type="kernel",
                    implementation=_name,
                    input_shape=(input_size,),
                    output_shape=(output_size,),
                )
            )
        )

    register_name(name)
    yield impl


_default_type_list_for_reordering = (
    (
        Conv1d,
        MaxPool1d,
        BatchNorm1d,
        Binarize,
    ),
)


class Torch2IrConverter:
    def __init__(
        self,
        leafs=(Binarize,),
        block_matcher_factory: Callable[
            [Module, torch.fx.Graph], LutronBlockMatcher
        ] = _DefaultLutronBlockMatcher,
        type_lists_for_reordering: tuple[
            tuple[str | Callable | type, ...], ...
        ] = _default_type_list_for_reordering,
    ):
        self._leafs = leafs + (LutronModule,)
        self._t = LeafTracer(self._leafs)
        self._g: torch.fx.Graph = torch.fx.Graph()
        self._model: Module = torch.nn.Identity()
        self._input_shape: tuple[int, int, int] = (1, 1, 1)
        self.block_matcher_factory = block_matcher_factory
        self.type_lists_for_reordering = type_lists_for_reordering

    def convert(
        self, model: Module, input_shape: tuple[int, int, int]
    ) -> dict[str, Implementation]:
        self._model = model
        self._input_shape = input_shape
        self._remove_identities()
        self._reorder_conv_blocks()
        self._generate_lutron_blocks()
        result = {}
        for impl in self._generate_ir():
            name = impl.name
            result[name] = impl
        return result

    def _retrace(self) -> None:
        self._g = self._t.trace(self._model)

    @property
    def leafs(self) -> tuple[type, ...]:
        return self._leafs

    @leafs.setter
    def leafs(self, value: tuple[type, ...]):
        self._leafs = value + (LutronModule,)
        self._t = LeafTracer(self._leafs)

    def _remove_identities(self) -> None:
        self._retrace()
        self._model = remove_identities(self._model, graph=self._g)

    def _reorder_conv_blocks(self) -> None:
        self._retrace()
        self._model = reorder_conv_blocks(
            self._model, graph=self._g, type_lists=self.type_lists_for_reordering
        )

    def _generate_lutron_blocks(self):
        self._retrace()
        self._model = autogenerate_lutron_blocks(
            self._model,
            input_shape=self._input_shape,
            graph=self._g,
            block_matcher_factory=self.block_matcher_factory,
        )

    def _generate_ir(self) -> Iterator[_Implementation[Node, Edge]]:
        self._retrace()
        model = self._model
        input_shape = self._input_shape
        g = self._g
        modules = dict(model.named_modules())

        top = Implementation("network", "network")
        yield top
        node_name_mapping = {}
        impl_name_mapping = {"": ""}

        def make_vhdl_name(name: str):  # TODO: move this to ir2vhdl
            if len(name) == 0:
                return name
            if name[0].isdecimal():
                name = f"_{name}"
            if name[0] in (".", "_"):
                name = f"XdX{name[1:]}"
            name = name.replace(".", "_")
            return name

        for n in g.nodes:
            if n.op == "placeholder":
                node_name_mapping[n.name] = "input"
                top.input(size=input_shape[2], channels=input_shape[1])
            if n.op == "output":
                node_name_mapping[n.name] = "output"
                output_predecessors = n.all_input_nodes
                if len(output_predecessors) > 1:
                    raise ValueError(
                        "more than one return value is not supported, received {}",
                        n.all_input_nodes,
                    )
                predecessor = output_predecessors[0]
                pre_module: LutronModule = cast(LutronModule, modules[predecessor.name])
                top.output(
                    size=pre_module.filter_parameters.output_size,
                    channels=pre_module.filter_parameters.out_channels,
                )
            if n.op == "call_module":
                node_name = make_vhdl_name(n.name)
                node_name_mapping[n.name] = node_name
                target = cast(str, n.target)
                impl_name = make_vhdl_name(target)

                impl_name_mapping[target] = impl_name
                module = modules[target]
                if isinstance(module, LutronModule):
                    top.lutron_filter(
                        name=node_name,
                        module=module,
                        implementation=impl_name,
                    )
                else:
                    raise NotImplementedError(
                        f"trying to translate {n.target} of type"
                        f" {type(module)}, but only supporting lutron filters currently"
                    )

        for n in g.nodes:
            t_name = node_name_mapping[n.name]
            for u in n.users:
                u_name = node_name_mapping[u.name]
                t_n = top.nodes[t_name]
                t_u = top.nodes[u_name]
                if t_u.input_shape.width != t_n.output_shape.width:
                    raise ValueError(
                        "nodes have incompatible data widths src: {}, sink: {}",
                        t_n,
                        t_u,
                    )
                top.add_edge(
                    Edge(dict(src=t_name, sink=u_name, src_sink_indices=tuple()))
                )

        yield top

        registry: dict[str, _Implementation[Node, Edge] | None] = {"network": top}
        for n in g.nodes:
            if n.op == "call_module":
                module = modules[cast(str, n.target)]
                name = n.target
                if hasattr(module, "filter_parameters"):
                    name = cast(str, impl_name_mapping[cast(str, name)])
                    yield from add_lutron_filter_impl(name, module, registry)
