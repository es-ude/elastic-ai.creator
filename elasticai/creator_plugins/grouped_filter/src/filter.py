from collections.abc import Callable
from itertools import chain
from typing import ParamSpec

import elasticai.creator.hdl_ir as ir
from elasticai.creator.hdl_ir import IrFactory, Node, Shape
from elasticai.creator.ir import Registry as _Registry

from .filter_params import FilterParameters
from .index_generators import GroupedFilterIndexGenerator

factory = IrFactory()

P = ParamSpec("P")

type InDGraph = ir.DataGraph[ir.Node, ir.Edge]
type InRegistry = ir.Registry
type Registry = ir.Registry


def append_counter_suffix_before_construction[**P](
    fn: Callable[P, Node],
) -> Callable[P, Node]:
    counters: dict[str, int] = {}

    def construct(*args: P.args, **kwargs: P.kwargs) -> Node:
        nonlocal counters
        if "name" in kwargs:
            name = kwargs["name"]
            kwargs.pop("name")
            if not isinstance(name, str):
                raise TypeError("expected `name` to be of type str")
        elif isinstance(args[0], str):
            name = args[0]
        else:
            raise TypeError("missing positional arg `name`")

        count = counters.get(name, 0)
        counters[name] = count + 1

        new_name = f"{name}_i{count}"
        args = tuple(chain((new_name,), args[1:]))  # type: ignore
        node = fn(*args, **kwargs)
        return node

    return construct


def grouped_filter(
    impl: InDGraph, registry: InRegistry
) -> tuple[ir.DataGraph, InRegistry]:
    nc = append_counter_suffix_before_construction(factory.node)
    result = factory.graph(type="unclocked_combinatorial")
    params = FilterParameters.from_dict(impl.attributes["filter_parameters"])
    result = result.add_node(
        factory.node(
            "input",
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

    # need to reverse kernels, as this implementation assumes that higher
    # id on wire means earlier in time or higher in channel
    kernels = reversed(kernels)
    for kernel, wires_per_step in zip(kernels, g.as_tuple_by_groups()):
        wires = wires_per_step[0]
        node = nc(
            kernel,
            type="unclocked_combinatorial",
            implementation=kernel,
            input_shape=Shape(
                len(wires),
            ),
            output_shape=Shape(
                params.out_channels_per_group,
            ),
        )
        result = result.add_node(node)
        result = result.add_edges(
            factory.edge(
                "input",
                node.name,
                src_dst_indices=tuple(zip(wires, range(len(wires)))),
            ),
            factory.edge(
                node.name,
                "output",
                src_dst_indices=(
                    f"range(0, {params.out_channels_per_group})",
                    f"range({output_offset}, {output_offset + params.out_channels_per_group})",
                ),
            ),
        )
        output_offset += params.out_channels_per_group

    result = result.add_node(
        factory.node(
            "output",
            type="output",
            implementation="",
            input_shape=Shape(params.out_channels, 1),
            output_shape=Shape(params.out_channels, 1),
        )
    )
    return result, _Registry()
