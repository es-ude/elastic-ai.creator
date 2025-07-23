import warnings

from elasticai.creator.ir2vhdl import Instance, VhdlNode

from .clocked_combinatorial import ClockedInstance
from .node_factory import (
    InstanceFactoryForCombinatorial,
)


@InstanceFactoryForCombinatorial.register
def shift_register(node: VhdlNode) -> Instance:
    return ShiftRegister(node)


def _check_input_output_shape_compatibility(node):
    input_shape, output_shape = node.input_shape, node.output_shape
    if output_shape.size() % input_shape.size() != 0:
        raise ValueError(
            "Found incompatible input output shapes for shift_register. Total output size has to be an integer multiple of total input size, but found output={output} and input={input}.".format(
                output=output_shape, input=input_shape
            )
        )
    if output_shape.depth != input_shape.depth:
        warnings.warn(
            'Detected mismatching input output shapes for shift_register for node "{}". Width of output and input shape should usually be equal, but found output={} and input={}.'.format(
                node.name, output_shape, input_shape
            ),
            stacklevel=3,
        )


class ShiftRegister(ClockedInstance):
    _logic_signals_with_default_suffix = ("valid_in", "valid_out")

    def __init__(self, node: VhdlNode):
        _check_input_output_shape_compatibility(node)
        data_width = node.input_shape.size()
        output_width = node.output_shape.size()
        if output_width % data_width != 0:
            raise ValueError(
                "incompatible shapes input: {}  output: {}".format(
                    node.input_shape, node.output_shape
                )
            )
        num_points = output_width // data_width
        super().__init__(
            node,
            input_width=data_width,
            output_width=output_width,
            generic_map={"data_width": data_width, "num_points": num_points},
        )
