from .ir_data_meta import IrDataMeta
from .attributes_descriptor import AttributesDescriptor
from .attribute import Attribute
from collections.abc import Callable
from typing import TypeVar

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


def ir_data_class(cls) -> Callable[[dict[str, Attribute]], IrData]:
    return type(cls.__name__, (cls, IrData), {})
