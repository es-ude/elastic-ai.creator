from typing import overload

from .abstract_ir_data import AbstractIRData
from .abstract_ir_data.mandatory_field import MandatoryField
from .attribute import AttributeT


class Node(AbstractIRData):
    name: MandatoryField[str] = MandatoryField()
    type: MandatoryField[str] = MandatoryField()

    @classmethod
    @overload
    def new(cls, name: str, type: str, attributes: dict[str, AttributeT]) -> "Node": ...

    @classmethod
    @overload
    def new(cls, name: str, type: str) -> "Node": ...

    @classmethod
    def new(cls, *args, **kwargs) -> "Node":
        return cls._do_new(*args, **kwargs)

    def __hash__(self):
        return hash((self.name, self.type))
