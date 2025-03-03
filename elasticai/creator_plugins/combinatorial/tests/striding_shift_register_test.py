from elasticai.creator.ir2vhdl import Shape, VhdlNode
from elasticai.creator_plugins.combinatorial.vhdl_nodes.striding_shift_register import (
    striding_shift_register,
)

from .shift_register_test import extract_code_section


def test_shift_register_converts_depth_and_width_to_correct_generics():
    conv0_channels = 2
    conv1_kernel_size = 3
    conv0_out_shape = Shape(conv0_channels, 1)
    conv1_in_shape = Shape(conv0_channels, conv1_kernel_size)
    stride = 2
    n = VhdlNode(
        name="sr0",
        data=dict(
            type="shift_register",
            implementation="",
            input_shape=conv0_out_shape.to_tuple(),
            output_shape=conv1_in_shape.to_tuple(),
            stride=stride,
        ),
    )
    sr = striding_shift_register(n)
    entity = tuple(extract_code_section(sr.instantiate(), start="generic", end=")"))
    entity = set(line.strip(",") for line in entity)
    expected = {
        f"DATA_WIDTH => {conv0_channels}",
        f"NUM_POINTS => {conv1_in_shape.width}",
        f"STRIDE => {stride}",
    }
    assert entity == expected
