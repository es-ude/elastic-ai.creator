import elasticai.creator.ir.ir_v2 as ir
from elasticai.creator.ir2vhdl import Shape, factory

from ..clocked_combinatorial import clocked_combinatorial


def test_using_two_clocked_components_connects_their_valid_signals():
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
    lines = set(l.lstrip() for l in lines)
    for line in lines:
        if line.startswith("valid"):
            print(line)
    valid_ins = [line for line in lines if line.startswith("valid_in")]
    assert "valid_in_shift_register <= valid_out_sliding_window;" in valid_ins
