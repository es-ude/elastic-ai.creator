from collections.abc import Sequence

from elasticai.creator import ir
from elasticai.creator.ir2vhdl import (
    Code,
    Ir2Vhdl,
    IrFactory,
    PluginLoader,
)


def _assert_line_structure(code: Code, expected_name: str) -> None:
    name, lines = code
    assert name == expected_name
    assert isinstance(lines, Sequence)
    assert all(isinstance(line, str) for line in lines)


def _translate_single_graph(graph):
    translate = Ir2Vhdl()
    PluginLoader(translate).load_from_package("skeleton")
    return tuple(translate(graph, ir.Registry(())))


def test_skeleton_type_handler_output_structure() -> None:
    graph = IrFactory().graph(
        ir.attribute(
            generic_map={
                "DATA_IN_WIDTH": "8",
                "DATA_IN_DEPTH": "4",
                "DATA_OUT_WIDTH": "8",
                "DATA_OUT_DEPTH": "4",
            }
        ),
        type="skeleton",
        name="skeleton",
    )
    generated = dict(_translate_single_graph(graph))
    _assert_line_structure(("skeleton.vhd", generated["skeleton.vhd"]), "skeleton.vhd")
