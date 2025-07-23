from collections.abc import Callable, Iterable
from typing import ParamSpec

import elasticai.creator.plugin as _pl
from elasticai.creator.function_utils import FunctionDecorator
from elasticai.creator.graph import dfs_iter
from elasticai.creator.ir import RequiredField
from elasticai.creator.ir2vhdl import (
    Edge,
    Implementation,
    LoweringPass,
    Shape,
    edge,
    vhdl_node,
)
from elasticai.creator.ir2vhdl import (
    VhdlNode as Node,
)
from elasticai.creator_plugins.grouped_filter import FilterParameters

P = ParamSpec("P")


def _type_handler_fn(
    name: str, fn: Callable[[Implementation], Implementation]
) -> _pl.PluginSymbolFn[LoweringPass, [Implementation], Implementation]:
    def load_into(lower: LoweringPass[Implementation, Implementation]):
        lower.register(name, fn)

    return _pl.make_plugin_symbol(load_into, fn)


def _iterable_type_handler_fn(
    name: str, fn: Callable[[Implementation], Iterable[Implementation]]
) -> _pl.PluginSymbolFn[LoweringPass, [Implementation], Iterable[Implementation]]:
    def load_into(lower: LoweringPass[Implementation, Implementation]):
        lower.register_iterable(name, fn)

    return _pl.make_plugin_symbol(load_into, fn)


_type_handler = FunctionDecorator(_type_handler_fn)
_iterable_type_handler = FunctionDecorator(_iterable_type_handler_fn)


class _FilterNode(Node):
    filter_parameters: RequiredField[dict, FilterParameters] = RequiredField(
        set_convert=lambda x: x.as_dict(), get_convert=FilterParameters.from_dict
    )


def append_counter_suffix_before_construction(
    fn: Callable[P, Node],
) -> Callable[P, Node]:
    counters: dict[str, int] = {}

    def construct(*args: P.args, **kwargs: P.kwargs) -> Node:
        nonlocal counters
        if "name" in kwargs:
            name = kwargs["name"]
            if not isinstance(name, str):
                raise TypeError("expected `name` to be of type str")
        elif isinstance(args[0], str):
            name = args[0]
        else:
            raise TypeError("missing positional arg `name`")

        count = counters.get(name, 0)
        counters[name] = count + 1
        kwargs["name"] = f"{name}_i{count}"
        node = fn(*args, **kwargs)
        return node

    return construct


class _Sequential:
    def __init__(self, name: str):
        self._impl: Implementation[Node, Edge] = Implementation(
            name=name,
            type="clocked_combinatorial",
        )
        self._last_node: Node | None = None
        self._last_filter_parameters: None | FilterParameters = None
        self._counting_node_constructor = append_counter_suffix_before_construction(
            vhdl_node
        )
        self._num_registers: int = 0
        self._last_stride = 1

    def add_input(self, shape: Shape):
        self._impl.add_node(
            vhdl_node(
                name="input",
                type="input",
                implementation="",
                output_shape=shape,
                input_shape=shape,
            )
        )
        self._last_node = self._impl.nodes["input"]
        self._update_last_filter_params()

    def _need_shift_register(self, params: FilterParameters) -> bool:
        if self._last_node is None:
            return False
        consuming_more_than_last_node_produces = (
            self._last_node.output_shape.width < params.kernel_size
        )
        return consuming_more_than_last_node_produces

    def _need_sliding_window(self, params: FilterParameters) -> bool:
        if self._last_node is None:
            return False
        consuming_less_than_last_node_produces = (
            self._last_node.output_shape.width > params.kernel_size
        )
        return consuming_less_than_last_node_produces

    def filter(self, n: Node):
        node = _FilterNode(n.name, n.data)
        attributes = n.attributes
        params = node.filter_parameters
        if "top_stride" not in self._impl.attributes:
            self._impl.data.update(
                {
                    "top_stride": params.in_channels * params.stride,
                }
            )
        if self._need_shift_register(params):
            old_params = self._last_filter_parameters
            self.strided_shift_register(
                output_shape=(
                    params.in_channels,
                    params.kernel_size,
                ),
                stride=old_params.stride,
            )
        elif self._need_sliding_window(params):
            self._sliding_window(params)
        elif self._last_node is not None:
            pass
        else:
            raise ValueError("expected last node to be not None")
        self._append_node(
            name=n.name,
            type="unclocked_combinatorial",
            implementation=n.implementation,
            output_shape=Shape(attributes["filter_parameters"]["out_channels"], 1),
            attributes=attributes,
            node_fn=vhdl_node,
        )

    def _sliding_window(self, params: FilterParameters) -> None:
        self._append_static(
            "sliding_window",
            "sliding_window",
            output_shape=Shape(
                params.in_channels,
                params.kernel_size,
            ),
        )

    def _update_last_filter_params(self):
        new_node = self._last_node
        if "filter_parameters" in new_node.data:
            params = _FilterNode(new_node.name, new_node.data).filter_parameters

            self._last_filter_parameters is not None
            if params.kernel_size == 1:
                self._last_filter_parameters = FilterParameters(
                    kernel_size=params.kernel_size,
                    in_channels=params.in_channels,
                    out_channels=params.out_channels,
                    groups=params.groups,
                    stride=self._last_filter_parameters.stride * params.stride,
                    input_size=params.input_size,
                    output_size=params.output_size,
                )
            else:
                self._last_filter_parameters = params
        elif self._last_filter_parameters is None:
            self._last_filter_parameters = FilterParameters(
                kernel_size=1,
                in_channels=new_node.input_shape.depth,
                out_channels=new_node.output_shape.depth,
            )

    def _append_node(
        self,
        name: str,
        output_shape: Shape,
        type: str,
        implementation: str,
        node_fn,
        attributes=None,
    ):
        old_node = self._last_node
        if old_node is None:
            raise Exception("no input node")
        input_shape = old_node.output_shape
        new_node = node_fn(
            name=name,
            input_shape=input_shape,
            output_shape=output_shape,
            type=type,
            implementation=implementation,
            attributes=attributes if attributes is not None else {},
        )
        self._impl.add_node(new_node)
        self._last_node = new_node
        self._update_last_filter_params()
        if old_node is not None:
            self._impl.add_edge(
                edge(
                    src=old_node.name,
                    dst=new_node.name,
                    src_dst_indices=tuple(),
                )
            )

    def _append_static(
        self, name: str, implementation: str, output_shape: Shape, **kwargs
    ):
        self._append_node(
            name=name,
            output_shape=output_shape,
            implementation=implementation,
            type=implementation,
            attributes=kwargs,
            node_fn=self._counting_node_constructor,
        )

    def shift_register(self, name: str, output_shape: Shape):
        self._append_static(
            name=name,
            output_shape=output_shape,
            implementation="shift_register",
        )

    def strided_shift_register(self, output_shape: tuple[int, int], stride: int):
        self._num_registers += 1
        if stride > 1:
            self._append_static(
                name="striding_shift_register",
                output_shape=Shape(*output_shape),
                implementation="striding_shift_register",
                generic_map=dict(stride=stride),
            )
        else:
            self._append_static(
                name="shift_register",
                output_shape=Shape(*output_shape),
                implementation="shift_register",
                generic_map=dict(),
            )

    def input(self, impl: Implementation, input_node: str) -> None:
        input_shape = self._determine_required_input_shape(impl, input_node)
        self._impl.data["top_kernel_size"] = input_shape.size()
        self.add_input(input_shape)

    def _determine_required_input_shape(
        self, impl: Implementation, input_node: str
    ) -> Shape:
        first_node_after_input = tuple(impl.successors(input_node).values())[0]
        match first_node_after_input.type:
            case "filter":
                n = _FilterNode(
                    first_node_after_input.name, first_node_after_input.data
                )
                return Shape(
                    n.filter_parameters.in_channels, n.filter_parameters.kernel_size
                )
            case _:
                return impl.nodes[input_node].output_shape

    def set_runtime_input_shape(self, s: Shape) -> None:
        self._impl.data["runtime_input_shape"] = s.to_tuple()

    def set_runtime_output_shape(self, s: Shape) -> None:
        self._impl.data["runtime_output_shape"] = s.to_tuple()

    def get_impl(self) -> Implementation:
        return self._impl

    def output(self, n):
        if self._last_node.output_shape.width < n.input_shape.width:
            self.shift_register("shift_register", n.output_shape)
        self._append_node(
            name="output",
            output_shape=self._last_node.output_shape,
            type="output",
            implementation="",
            node_fn=vhdl_node,
        )


@_type_handler
def sequential(impl: Implementation) -> Implementation:
    seq = _Sequential(impl.name)

    def iter_nodes():
        for input_node in impl.nodes.values():
            if input_node.type == "input":
                break

        def iterator():
            yield from dfs_iter(impl.successors, input_node.name)

        return impl.get_node_mapping(iterator)

    for n in iter_nodes().values():
        match n.type:
            case "filter":
                seq.filter(n)
            case "input":
                seq.set_runtime_input_shape(n.input_shape)
                seq.input(impl, n.name)
            case "output":
                seq.set_runtime_output_shape(n.output_shape)
                seq.output(n)
            case _:
                raise Exception(
                    f"Can't handle unknown type {n.type} during generation of time multiplexed sequential"
                )

    return seq.get_impl()


@_iterable_type_handler
def network(impl: Implementation) -> Iterable[Implementation]:
    network = sequential(impl)
    network.attributes["top_kernel_size"]
    network.attributes["top_stride"]
    input_shape = network.nodes["input"].input_shape
    output_shape = network.nodes["output"].output_shape
    input_width, input_depth = input_shape
    output_width, output_depth = output_shape

    skeleton: Implementation = Implementation(name="skeleton", type="skeleton")
    skeleton.data["generic_map"] = {
        "DATA_IN_WIDTH": str(input_width),
        "DATA_IN_DEPTH": str(input_depth),
        "DATA_OUT_WIDTH": str(output_width),
        "DATA_OUT_DEPTH": str(output_depth),
    }

    buffered_network_wrapper: Implementation = Implementation(
        name="buffered_network_wrapper",
        type="buffered_network_wrapper",
    )
    return network, skeleton, buffered_network_wrapper
