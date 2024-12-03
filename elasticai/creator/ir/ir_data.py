from collections.abc import Callable
from typing import TypeVar

from .attribute import Attribute
from .attributes_descriptor import AttributesDescriptor
from .ir_data_meta import IrDataMeta

Self = TypeVar("Self", bound="IrData")


class IrData(metaclass=IrDataMeta, create_init=False):
    _fields: dict[str, type]  # only here for type checkers

    attributes: AttributesDescriptor = AttributesDescriptor()

    def __init__(self: Self, data: dict[str, Attribute]) -> None:
        self.data = data

    def get_missing_required_fields(self) -> dict[str, type]:
        missing: dict[str, type] = {}
        for name, _type in self._fields.items():
            if name not in self.data:
                missing[name] = _type
        return missing

    def __repr__(self) -> str:
        return f"{self.__class__} ({self.data})"

    def __eq__(self, o: object) -> bool:
        if self is o:
            return True
        if isinstance(o, self.__class__):
            return o.data == self.data
        return False


def ir_data_class(cls) -> Callable[[dict[str, Attribute]], IrData]:
    return type(cls.__name__, (cls, IrData), {})
