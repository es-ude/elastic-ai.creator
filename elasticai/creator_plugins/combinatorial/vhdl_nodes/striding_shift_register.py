from elasticai.creator.ir2vhdl import VhdlNode

from .node_factory import (
    InstanceFactoryForCombinatorial,
)
from .shift_register import ShiftRegister


@InstanceFactoryForCombinatorial.register
def striding_shift_register(node: VhdlNode):
    return StridingShiftRegister(node)


class StridingShiftRegister(ShiftRegister):
    def __init__(
        self,
        node: VhdlNode,
    ):
        super().__init__(node)
        if (
            "generic_map" in node.attributes  # pyright: ignore
            and "stride" in node.attributes["generic_map"]  # pyright: ignore
        ):
            stride = node.attributes["generic_map"]["stride"]  # pyright: ignore
        else:
            stride = node.attributes["stride"]  # pyright: ignore
        self._generics["stride"] = stride
