from collections.abc import Callable

from .attribute import Attribute
from .attributes_descriptor import AttributesDescriptor
from .ir_data_meta import IrDataMeta


class IrData(metaclass=IrDataMeta, create_init=False):
    """Convenience class for creating new Ir data classes.

    To create a new Ir data class, inherit from `IrData` and define your required fields.
    Supported field types are

    - xref:elasticai-creator:api:elasticai.creator.ir.required_field.adoc#RequiredField[`RequiredField`]
    - xref:elasticai-creator:api:elasticai.creator.ir.required_field.adoc#ReadOnlyField[`ReadOnlyField`]
    - xref:elasticai-creator:api:elasticai.creator.ir.required_field.adoc#SimpleField[`SimpleField`]

    Examples:

    We can define a new class `MyNewIrData` with the two required fields `name` and `length` like this:

    ```python
    from elasticai.creator.ir import IrData, SimpleRequiredField, RequiredField

    class MyNewIrData(IrData):
        name: SimpleRequiredField[str] = SimpleRequiredField()  # <1>
        length: RequiredField[str, int] = RequiredField(str, int)  # <2>

    d = MyNewIrData({'name': 'my name', 'length': '12'})
    d.name = 'new name'  # <3>
    d.length += 3  # <4>
    ```

    1. We define a simple required field. Data will be stored as string and can be accessed as string.
    2. Opposed to 1. this field will store data as a string but every read or write access will convert
    length to an integer. The constructor `str` will be called when writing a value to the field and `int`
    will be called on read.
    3. `'new name'` will be stored as is inside `d`'s data dictionary.
    4  `'12'` will be converted to `12` before adding `3` to it, the subsequent write operation will convert `15`
    to the string `'15'` again.

    IMPORTANT: Currently we assume that calling the conversion functions for a field will not result in any side effects.
    IMPORTANT: While it is ok to extend the available field types by inheritance the *type annotations*
      still need to refer to one of the provided field types, because they need to be analyzed by the metaclass.
    """

    _fields: dict[str, type]  # only here for type checkers

    attributes: AttributesDescriptor = AttributesDescriptor()

    def __init__(self, data: dict[str, Attribute]) -> None:
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
