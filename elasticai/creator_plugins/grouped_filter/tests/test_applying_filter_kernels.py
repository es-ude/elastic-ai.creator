import elasticai.creator.ir.ir_v2 as ir
from elasticai.creator.ir2vhdl import DataGraph, Edge, Shape, factory
from elasticai.creator_plugins.grouped_filter import (
    FilterParameters,
)
from elasticai.creator_plugins.grouped_filter import (
    grouped_filter as grouped_filter,
)


def high_level_ir(groups: int) -> DataGraph:
    p = FilterParameters(
        kernel_size=2, in_channels=2, out_channels=4, input_size=10, groups=groups
    )
    hl_ir = factory.graph(
        ir.attribute(
            {
                "filter_parameters": p.as_dict(),
                "kernel_per_group": tuple(f"kernel_{i}" for i in range(groups)),
            }
        ),
        type="kernel_filter",
    ).add_nodes(
        factory.node(
            "input",
            type="input",
            input_shape=Shape(p.in_channels, p.input_size),
            output_shape=Shape(p.out_channels, p.num_steps),
        ),
        factory.node(
            "output",
            type="output",
            input_shape=Shape(p.in_channels, p.input_size),
            output_shape=Shape(p.out_channels, p.num_steps),
        ),
        factory.node(
            "kernel_0",
            input_shape=Shape(
                p.in_channels * p.kernel_size,
            ),
            output_shape=Shape(
                p.out_channels,
            ),
            type="kernel",
            implementation="kernel_0",
        ),
    )
    return hl_ir


class TestFilterWithSingleGroup:
    def test_has_correct_edges(self) -> None:
        hl_ir = high_level_ir(1)
        p = FilterParameters.from_dict(hl_ir.attributes["filter_parameters"])
        expected_edges: set[Edge] = set()
        expected_edges.add(
            factory.edge(
                "input",
                "kernel_0_i0",
                src_dst_indices=tuple(
                    zip(
                        range(p.in_channels * p.kernel_size),
                        range(p.in_channels * p.kernel_size),
                    )
                ),
            )
        )
        expected_edges.add(
            factory.edge(
                "kernel_0_i0",
                "output",
                src_dst_indices=(
                    f"range(0, {p.out_channels})",
                    f"range(0, {p.out_channels})",
                ),
            )
        )
        actual_ir, _ = grouped_filter(hl_ir, ir.Registry())
        assert set(actual_ir.edges.values()) == expected_edges


class TestFilterWithTwoGroups:
    def test_has_correct_edges(self) -> None:
        """
        Note that the module interprets higher src_dst index as either earlier in time
        or more significant bit of single channel or channel with lower id.
        """
        hl_ir = high_level_ir(2)
        FilterParameters.from_dict(hl_ir.attributes["filter_parameters"])
        expected_edges = (
            factory.edge(
                "input",
                "kernel_0_i0",
                src_dst_indices=(
                    (0, 0),
                    (2, 1),
                ),
            ),
            factory.edge(
                "input",
                "kernel_1_i0",
                src_dst_indices=((1, 0), (3, 1)),
            ),
            factory.edge(
                "kernel_0_i0",
                "output",
                src_dst_indices=("range(0, 2)", "range(0, 2)"),
            ),
            factory.edge(
                "kernel_1_i0",
                "output",
                src_dst_indices=("range(0, 2)", "range(2, 4)"),
            ),
        )
        actual_ir, _ = grouped_filter(hl_ir, ir.Registry())
        assert set(actual_ir.edges.values()) == set(expected_edges)
