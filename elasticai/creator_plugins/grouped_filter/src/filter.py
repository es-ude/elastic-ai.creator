from collections.abc import Callable
from typing import ParamSpec

import elasticai.creator.plugin as _pl
from elasticai.creator.function_utils import FunctionDecorator
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

from .filter_params import FilterParameters
from .index_generators import GroupedFilterIndexGenerator

P = ParamSpec("P")


def _type_handler_fn(
    name: str, fn: Callable[[Implementation], Implementation]
) -> _pl.PluginSymbol:
    def load_into(lower: LoweringPass[Implementation, Implementation]):
        lower.register(name, fn)

    return _pl.make_plugin_symbol(load_into, fn)


_type_handler = FunctionDecorator(_type_handler_fn)


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


@_type_handler
def grouped_filter(impl: Implementation) -> Implementation:
    nc: Callable = append_counter_suffix_before_construction(vhdl_node)
    result: Implementation[Node, Edge] = Implementation(
        name=impl.name, type="unclocked_combinatorial", data={}
    )
    params = FilterParameters.from_dict(impl.attributes["filter_parameters"])
    result.add_node(
        vhdl_node(
            name="input",
            type="input",
            implementation="",
            input_shape=Shape(params.in_channels, params.kernel_size),
            output_shape=Shape(params.in_channels, params.kernel_size),
        )
    )
    kernels = impl.attributes["kernel_per_group"]
    if len(kernels) != params.groups:
        raise ValueError(
            "number of kernels per group should match the number of groups but found {}  and {}".format(
                len(kernels), params.groups
            )
        )
    g = GroupedFilterIndexGenerator(
        params=FilterParameters(
            kernel_size=params.kernel_size,
            in_channels=params.in_channels,
            out_channels=params.out_channels,
            stride=1,
            input_size=params.kernel_size,
            groups=params.groups,
        )
    )
    output_offset = 0
    for kernel, wires_per_step in zip(kernels, g.as_tuple_by_groups()):
        wires = wires_per_step[0]
        node = nc(
            name=kernel,
            type="unclocked_combinatorial",
            implementation=kernel,
            input_shape=Shape(
                len(wires),
            ),
            output_shape=Shape(
                params.out_channels_per_group,
            ),
        )
        result.add_node(node)
        result.add_edge(
            edge(
                src="input",
                dst=node.name,
                src_dst_indices=tuple(zip(wires, range(len(wires)))),
            )
        )
        result.add_edge(
            edge(
                src=node.name,
                dst="output",
                src_dst_indices=(
                    f"range(0, {params.out_channels_per_group})",
                    f"range({output_offset}, {output_offset + params.out_channels_per_group})",
                ),
            ),
        )
        output_offset += params.out_channels_per_group

    result.add_node(
        vhdl_node(
            name="output",
            type="output",
            implementation="",
            input_shape=Shape(params.out_channels, 1),
            output_shape=Shape(params.out_channels, 1),
        )
    )
    return result
