import re
from collections.abc import Iterable

import pytest

import elasticai.creator.ir.ir_v2 as ir
from elasticai.creator.ir2vhdl import Node as VhdlNode
from elasticai.creator.ir2vhdl import Shape, factory
from elasticai.creator.ir2vhdl.language import Instance

from ..vhdl_nodes.node_factory import (
    InstanceFactoryForCombinatorial,
)


@pytest.fixture(scope="class")
def node(raw_node):
    return InstanceFactoryForCombinatorial(raw_node)


def new_node(
    name: str,
    type: str,
    implementation: str,
    input_shape: Shape,
    output_shape: Shape,
    attributes: dict | None = None,
) -> VhdlNode:
    if attributes is None:
        attributes = {}
    n = factory.node(
        name,
        ir.attribute(
            dict(**attributes),
        ),
        type=type,
        implementation=implementation,
        input_shape=input_shape,
        output_shape=output_shape,
    )

    return n


def _extract_assignments(starts_with: str, code: Iterable[str]) -> set[str]:
    assignments = set()
    extract = False
    for line in code:
        if line.startswith(starts_with):
            extract = True
        elif extract and "=>" in line:
            assignments.add(line.strip(","))
        elif extract:
            extract = False
    return assignments


class BaseVhdlNodeTest:
    def test_contains_all_signal_assignments(
        self, node: Instance, raw_node: VhdlNode, expected_assignments: set[str]
    ):
        code = list(line.strip() for line in node.instantiate())

        assert expected_assignments == _extract_assignments("port map (", code)

    def test_contains_all_generics(self, node: Instance, expected_generics: set[str]):
        code = (line.strip() for line in node.instantiate())
        assert expected_generics == _extract_assignments("generic", code)

    def test_first_line_is_correct(self, node: Instance, first_line: str):
        line = next(node.instantiate())
        assert first_line == line.strip()

    def test_can_define_signals(self, node, raw_node):
        signals = set(line for line in node.define_signals())
        expected = set(
            (
                f"signal d_in_{raw_node.name} : std_logic_vector({raw_node.input_shape.size()} - 1 downto 0) := (others => '0');",
                f"signal d_out_{raw_node.name} : std_logic_vector({raw_node.output_shape.size()} - 1 downto 0) := (others => '0');",
                f"signal src_valid_{raw_node.name} : std_logic := '0';",
                f"signal valid_{raw_node.name} : std_logic := '0';",
                f"signal dst_ready_{raw_node.name} : std_logic := '0';",
                f"signal ready_{raw_node.name} : std_logic := '0';",
            )
        )
        assert signals == expected


@pytest.mark.parametrize(
    ["in_shape", "out_shape", "stride", "expected"],
    [
        (Shape(1, 1), Shape(1, 4), 1, dict(DATA_WIDTH=1, NUM_POINTS=4, SKIP=1)),
        (Shape(1, 2), Shape(1, 4), 1, dict(DATA_WIDTH=2, NUM_POINTS=2, SKIP=1)),
        (Shape(2, 3), Shape(2, 6), 1, dict(DATA_WIDTH=6, NUM_POINTS=2, SKIP=1)),
        (Shape(2, 3), Shape(2, 6), 3, dict(DATA_WIDTH=6, NUM_POINTS=2, SKIP=3)),
    ],
)
def test_compute_correct_generic_map_for_shift_register(
    in_shape, out_shape, stride, expected
):
    _expected = {f"{k} => {v}" for k, v in expected.items()}
    result = {
        line.strip()
        for line in _extract_assignments(
            "generic map",
            InstanceFactoryForCombinatorial(
                new_node(
                    name="a",
                    type="shift_register",
                    implementation="shift_register",
                    input_shape=in_shape,
                    output_shape=out_shape,
                    attributes=dict(skip=stride),
                )
            ).instantiate(),
        )
    }
    assert result == _expected


class TestShiftRegister(BaseVhdlNodeTest):
    @pytest.fixture(scope="class")
    def raw_node(self):
        conv0_channels = 2
        conv1_kernel_size = 4
        conv0_out_shape = Shape(conv0_channels, 2)
        conv1_in_shape = Shape(conv0_channels, conv1_kernel_size)
        n = new_node(
            name="a",
            type="shift_register",
            implementation="impl",
            input_shape=conv0_out_shape,
            output_shape=conv1_in_shape,
        )
        return n

    @pytest.fixture(scope="class")
    def expected_generics(self) -> set[str]:
        """conv0 out shape is (2, 2) so we have data width of 4 and conv1 in shape is (2, 4) thus we have num points 2."""
        return {
            "NUM_POINTS => 2",
            "DATA_WIDTH => 4",
            "SKIP => 1",
        }

    @pytest.fixture(scope="class")
    def expected_assignments(self) -> set[str]:
        return {
            "clk => clk",
            "rst => rst",
            "en => en",
            "d_in => d_in_a",
            "d_out => d_out_a",
            "src_valid => src_valid_a",
            "valid => valid_a",
            "ready => ready_a",
            "dst_ready => dst_ready_a",
        }

    @pytest.fixture(scope="class")
    def first_line(self) -> str:
        return "a: entity work.impl(rtl)"


class TestSlidingWindow(BaseVhdlNodeTest):
    @pytest.fixture(scope="class")
    def raw_node(self):
        return new_node(
            name="a",
            type="sliding_window",
            input_shape=Shape(4, 2),
            output_shape=Shape(4, 1),
            implementation="impl",
            attributes={"stride": 2},
        )

    @pytest.fixture
    def expected_generics(self) -> set[str]:
        return {
            "STRIDE => 8",
            "INPUT_WIDTH => 8",
            "OUTPUT_WIDTH => 4",
        }

    @pytest.fixture
    def expected_assignments(self) -> set[str]:
        return {
            "clk => clk",
            "rst => rst",
            "en => en",
            "d_in => d_in_a",
            "d_out => d_out_a",
            "src_valid => src_valid_a",
            "valid => valid_a",
            "ready => ready_a",
            "dst_ready => dst_ready_a",
        }

    @pytest.fixture
    def first_line(self) -> str:
        return "a: entity work.impl(rtl)"

    def test_raise_error_for_incompatible_shapes(self):
        node_with_incompatible_shapes = new_node(
            name="a",
            type="sliding_window",
            input_shape=Shape(5, 2),
            output_shape=Shape(3, 2),
            implementation="impl",
            attributes={"stride": 2},
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Found incompatible input output shapes for sliding_window. Total input size has to be an integer multiple of total output size, but found output=Shape(depth=3, width=2) and input=Shape(depth=5, width=2)."
            ),
        ):
            InstanceFactoryForCombinatorial(node_with_incompatible_shapes).instantiate()

    def test_warn_about_technically_compatible_but_semantically_wrong_shapes(self):
        node_with_incompatible_shapes = new_node(
            name="a",
            type="sliding_window",
            input_shape=Shape(2, 4),
            output_shape=Shape(1, 2),
            implementation="impl",
            attributes={"stride": 2},
        )
        with pytest.warns(
            match=re.escape(
                'Detected mismatching input output shapes for sliding_window for node "a". Depth of output and input shape should usually be equal, but found output=Shape(depth=1, width=2) and input=Shape(depth=2, width=4).'
            ),
        ):
            InstanceFactoryForCombinatorial(node_with_incompatible_shapes).instantiate()
