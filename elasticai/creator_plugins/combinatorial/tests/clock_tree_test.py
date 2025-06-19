from elasticai.creator.ir2vhdl import Implementation, Shape, edge, vhdl_node

from ..clocked_combinatorial import clocked_combinatorial


def test_using_two_clocked_components_connects_their_valid_signals():
    impl = Implementation(name="test")
    impl.add_nodes(
        (
            vhdl_node(
                name="input",
                type="input",
                implementation="",
                input_shape=Shape(2, 4),
                output_shape=Shape(2, 4),
            ),
            vhdl_node(
                "sliding_window",
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
                output_shape=(1, 3),
            ),
            vhdl_node(
                "output",
                "output",
                "",
                input_shape=Shape(1, 3),
                output_shape=Shape(1, 3),
            ),
        )
    )
    impl.nodes["sliding_window"].data["generic_map"] = dict(stride=1)
    impl.add_edges(
        (
            edge("input", "sliding_window", []),
            edge("sliding_window", "conv", []),
            edge("conv", "shift_register", []),
            edge("shift_register", "output", []),
        )
    )
    result = clocked_combinatorial(impl)
    _, lines = result
    lines = set(l.lstrip() for l in lines)
    valid_ins = {line for line in lines if line.startswith("valid_in")}
    assert "valid_in_shift_register <= valid_out_sliding_window;" in valid_ins
