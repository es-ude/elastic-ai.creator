from elasticai.creator.ir2vhdl import Instance, VhdlNode

from .clocked_combinatorial import ClockedInstance
from .node_factory import InstanceFactoryForCombinatorial


@InstanceFactoryForCombinatorial.register
def shift_register(node: VhdlNode) -> Instance:
    return ShiftRegister(node)


class ShiftRegister(ClockedInstance):
    _logic_signals_with_default_suffix = ("valid_in", "valid_out")

    def __init__(self, node: VhdlNode):
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
