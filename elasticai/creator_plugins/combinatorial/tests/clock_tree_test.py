from collections.abc import Iterable, Sequence

import elasticai.creator.ir.ir_v2 as ir
import pytest

from elasticai.creator.ir2vhdl import Shape, factory
from elasticai.creator_plugins.combinatorial.clocked_combinatorial import (
    clocked_combinatorial,
)


def test_valid_path_assignments_are_present(code):
    assert "valid_input <= src_valid;" in code
    assert "src_valid_sliding_window <= valid_input;" in code

    assert "src_valid_shift_register <= valid_sliding_window;" in code
    assert "src_valid_output <= valid_shift_register;" in code
    assert "valid <= src_valid_output;" in code


def test_ready_path_assignments_are_present(code):
    assert "ready <= dst_ready_input;" in code
    assert "dst_ready_input <= ready_sliding_window;" in code
    assert "dst_ready_sliding_window <= ready_shift_register;" in code
    assert "dst_ready_shift_register <= ready_output;" in code
    assert "ready_output <= dst_ready;" in code


def test_all_assigned_signals_are_declared(code):
    directly_wired_ctrl_signals = (
        "rst",
        "en",
        "clk",
    )
    assert set(directly_wired_ctrl_signals) == collect_defined_signal(
        code
    ).intersection(directly_wired_ctrl_signals), "not all control signals are declared"
    defined_signals = collect_defined_signal(code)
    assigned_signals = collect_assignment_signals(code)
    assert assigned_signals == (defined_signals).difference(
        directly_wired_ctrl_signals
    ).intersection(assigned_signals), "some assigned signals were not declared"
    rhs_portmap_signals = collect_rhs_port_map_signals(code)
    assert rhs_portmap_signals == (defined_signals).intersection(rhs_portmap_signals)


@pytest.fixture
def code() -> Sequence[str]:
    impl = factory.graph(name="test")
    vhdl_node = factory.node
    edge = factory.edge
    impl = impl.add_nodes(
        vhdl_node(
            "input",
            type="input",
            implementation="",
            input_shape=Shape(2, 4),
            output_shape=Shape(2, 4),
        ),
        vhdl_node(
            "sliding_window",
            ir.attribute(stride=1),
            type="sliding_window",
            implementation="sliding_window",
            input_shape=Shape(2, 4),
            output_shape=Shape(2, 2),
        ),
        vhdl_node(
            "conv",
            type="unclocked_combinatorial",
            implementation="conv",
            input_shape=Shape(2, 4),
            output_shape=Shape(1, 1),
        ),
        vhdl_node(
            "shift_register",
            type="shift_register",
            implementation="shift_register",
            input_shape=Shape(1, 1),
            output_shape=Shape(1, 3),
        ),
        vhdl_node(
            "output",
            type="output",
            input_shape=Shape(1, 3),
            output_shape=Shape(1, 3),
        ),
    )
    impl = impl.add_edges(
        edge("input", "sliding_window"),
        edge("sliding_window", "conv"),
        edge("conv", "shift_register"),
        edge("shift_register", "output"),
    )
    result = clocked_combinatorial(impl, ir.Registry())
    _, lines = result
    lines = [line.strip() for line in lines]
    return lines


def collect_assignment_signals(code: Iterable[str]) -> set[str]:
    def split(assignment) -> tuple[str, str]:
        a, b = assignment.rstrip(";").split("<=")
        return a.strip(), b.strip()

    signals = set()
    assignments = (line for line in code if "<=" in line)
    for a, b in map(split, assignments):
        signals |= {a, b}
    return signals


def test_collect_assignments() -> None:
    code = ["a <= b;", "   c <= d  ;"]
    assert {"a", "b", "c", "d"} == collect_assignment_signals(code)


def collect_defined_signal(code: Iterable[str]) -> set[str]:
    defines = filter(lambda line: line.strip().startswith("signal"), code)
    signals = set()
    for line in defines:
        signals.add(line.split(":")[0].strip().removeprefix("signal").strip())

    return signals


def test_collecting_defined_signals() -> None:
    code = ["     signal a : some type and stuff;   "]
    assert {"a"} == collect_defined_signal(code)


def collect_rhs_port_map_signals(code: Iterable[str]) -> set[str]:
    inside_portmap = False
    signals = set()
    for line in code:
        if "port map" in line:
            inside_portmap = True
        elif inside_portmap:
            if "=>" in line:
                signal = line.split("=>")[1].strip(", ")
                signals.add(signal)
            if ");" in line:
                inside_portmap = False
    return signals


def test_collecting_rhs_portmap_signals() -> None:
    code = ["port map (", "   a => b,", "  c => d", ");", "x <= f", "r => t"]
    assert {"b", "d"} == collect_rhs_port_map_signals(code)
