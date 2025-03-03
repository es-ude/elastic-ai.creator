from collections.abc import Iterable

from elasticai.creator.ir2vhdl import Shape
from elasticai.creator.ir2vhdl import VhdlNode as Node
from elasticai.creator_plugins.combinatorial.vhdl_nodes import shift_register


def extract_code_section(lines: Iterable[str], start: str, end: str):
    section_is_relevant = False
    lines = map(str.strip, lines)
    lines = tuple(lines)
    for line in lines:
        if not section_is_relevant and line.startswith(start):
            section_is_relevant = True
        elif section_is_relevant and line.startswith(end):
            section_is_relevant = False
        elif section_is_relevant:
            yield line


def test_shift_register_converts_depth_and_width_to_correct_generics():
    conv0_channels = 2
    conv1_kernel_size = 3
    conv0_out_shape = Shape(conv0_channels, 1)
    conv1_in_shape = Shape(conv0_channels, conv1_kernel_size)
    n = Node(
        name="sr0",
        data=dict(
            type="shift_register",
            implementation="",
            input_shape=conv0_out_shape.to_tuple(),
            output_shape=conv1_in_shape.to_tuple(),
        ),
    )
    sr = shift_register(n)
    entity = tuple(extract_code_section(sr.instantiate(), start="generic", end=")"))
    entity = set(line.strip(",") for line in entity)
    expected = {
        f"DATA_WIDTH => {conv0_channels}",
        f"NUM_POINTS => {conv1_in_shape.width}",
    }
    assert entity == expected
