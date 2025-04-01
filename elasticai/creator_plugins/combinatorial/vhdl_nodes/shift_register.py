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
        num_points = node.output_shape.width
        output_width = node.output_shape.size()
        super().__init__(
            node,
            input_width=data_width,
            output_width=output_width,
            generic_map={"data_width": data_width, "num_points": num_points},
        )
