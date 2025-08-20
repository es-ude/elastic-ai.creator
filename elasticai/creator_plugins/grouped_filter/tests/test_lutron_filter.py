from elasticai.creator.ir2vhdl import (
    Edge,
    edge,
    vhdl_node,
)
from elasticai.creator.ir2vhdl import (
    Implementation as _Implementation,
)
from elasticai.creator_plugins.grouped_filter import (
    FilterParameters,
)
from elasticai.creator_plugins.grouped_filter import (
    grouped_filter as filter,
)


class Implementation(_Implementation):
    def input(self, shape: tuple[int, int]):
        self.add_node(
            vhdl_node(
                name="input",
                type="input",
                implementation="",
                input_shape=shape,
                output_shape=shape,
            )
        )

    def output(self, shape: tuple[int, int]):
        self.add_node(
            vhdl_node(
                name="output",
                type="output",
                implementation="",
                output_shape=shape,
                input_shape=shape,
            )
        )


def high_level_ir(groups: int) -> Implementation:
    p = FilterParameters(
        kernel_size=2, in_channels=2, out_channels=4, input_size=10, groups=groups
    )
    hl_ir = Implementation(
        name="filter",
        type="lutron_filter",
        data={
            "filter_parameters": p.as_dict(),
            "kernel_per_group": tuple(f"lutron_{i}" for i in range(groups)),
        },
    )
    hl_ir.input((p.in_channels, p.input_size))
    hl_ir.output((p.out_channels, p.num_steps))
    hl_ir.add_node(
        vhdl_node(
            name="lutron_0",
            input_shape=(p.in_channels * p.kernel_size,),
            output_shape=(p.out_channels,),
            type="lutron",
            implementation="lutron_0",
        )
    )
    return hl_ir


class TestLutronFilterWithSingleGroup:
    def test_has_correct_edges(self) -> None:
        hl_ir = high_level_ir(1)
        p = FilterParameters.from_dict(hl_ir.attributes["filter_parameters"])
        expected_edges: set[Edge] = set()
        expected_edges.add(
            edge(
                src="input",
                dst="lutron_0_i0",
                src_dst_indices=tuple(
                    zip(
                        range(p.in_channels * p.kernel_size),
                        range(p.in_channels * p.kernel_size),
                    )
                ),
            )
        )
        expected_edges.add(
            edge(
                src="lutron_0_i0",
                dst="output",
                src_dst_indices=(
                    f"range(0, {p.out_channels})",
                    f"range(0, {p.out_channels})",
                ),
            )
        )
        actual_ir = filter(hl_ir)
        assert set(actual_ir.edges.values()) == expected_edges


class TestLutronFilterWithTwoGroups:
    def test_has_correct_edges(self) -> None:
        hl_ir = high_level_ir(2)
        FilterParameters.from_dict(hl_ir.attributes["filter_parameters"])
        expected_edges: set[Edge] = set()
        expected_edges = expected_edges.union(
            (
                edge(
                    src="input",
                    dst="lutron_0_i0",
                    src_dst_indices=(
                        (0, 0),
                        (2, 1),
                    ),
                ),
                edge(
                    src="input",
                    dst="lutron_1_i0",
                    src_dst_indices=((1, 0), (3, 1)),
                ),
                edge(
                    src="lutron_0_i0",
                    dst="output",
                    src_dst_indices=("range(0, 2)", "range(0, 2)"),
                ),
                edge(
                    src="lutron_1_i0",
                    dst="output",
                    src_dst_indices=("range(0, 2)", "range(2, 4)"),
                ),
            )
        )
        actual_ir = filter(hl_ir)
        assert set(actual_ir.edges.values()) == expected_edges
