from collections.abc import Callable
from typing import Generic, TypeVar, cast

from typing_extensions import TypeIs

from ._has_data import HasData
from .attribute import Attribute

StoredT = TypeVar("StoredT", bound=Attribute)
VisibleT = TypeVar("VisibleT")


class RequiredField(Generic[StoredT, VisibleT]):
    """
    A __descriptor__ that designates a mandatory field of an abstract ir data class.
    The descriptor accesses the `data` dictionary of the owning abstract ir data object
    to read and write values. You can use the `set_convert` and `get_convert` functions
    to transform values during read/write accesses.
    That allows to keep dictionary of primitive (serializable) data types in memory,
    while still providing abstract ways to manipulate that data in complex ways.
    This is typically required when working with Nodes and Graphs to create new
    intermediate representations and transform one graph into another.

    NOTE: In most cases you want to ensure that the conditions
      - `x == get_convert(set_convert(x))`
      - `x == set_convert(get_convert(x))`
    hold for all applicable `x`.

    E.g.
    ```python
    >>> from elasticai.creator.ir import IrData
    >>> class A(IrData):
    ...    number: RequiredField[str, int] = RequiredField(set_transform=str, get_transform=int)
    >>> a = A({'number': "12"})
    >>> a.number = a.number + 3
    >>> a.data
    {'number': "15"}
    ```
    """

    __slots__ = ("set_convert", "get_convert", "name")

    def __init__(
        self,
        set_convert: Callable[[VisibleT], StoredT],
        get_convert: Callable[[StoredT], VisibleT],
    ):
        self.set_convert = set_convert
        self.get_convert = get_convert

    def __set_name__(self, owner: type[HasData], name: str) -> None:
        """
        IMPORTANT: do not remove owner even though it's not used
        see https://docs.python.org/3/reference/datamodel.html#descriptors for more information
        """
        self.name = name

    def __get__(self, instance: HasData, owner=None) -> VisibleT:
        """
        IMPORTANT: do not remove owner even though it's not used
        see https://docs.python.org/3/reference/datamodel.html#descriptors for more information
        """
        return self.get_convert(cast(StoredT, instance.data[self.name]))

    def __set__(self, instance: HasData, value: VisibleT) -> None:
        instance.data[self.name] = self.set_convert(value)


class SimpleRequiredField(Generic[StoredT]):
    slots = ("name",)

    def __set_name__(self, owner: type[HasData], name: str) -> None:
        self.name = name

    def __get__(self, instance: HasData, owner=None) -> StoredT:
        return cast(StoredT, instance.data[self.name])

    def __set__(self, instance: HasData, value: StoredT) -> None:
        instance.data[self.name] = value


class ReadOnlyField(Generic[StoredT, VisibleT]):
    slots = ("get_convert",)

    def __init__(self, get_convert: Callable[[StoredT], VisibleT]) -> None:
        self.get_convert = get_convert

    def __set_name__(self, owner: type[HasData], name: str) -> None:
        self.name = name

    def __get__(
        self, instance: HasData, owner: type[HasData] | None = None
    ) -> VisibleT:
        return self.get_convert(cast(StoredT, instance.data[self.name]))


_HasDataT = TypeVar("_HasDataT", bound=HasData)


class ReadOnlyMethodField(Generic[_HasDataT, StoredT, VisibleT]):
    __slots__ = ("get_convert", "name")

    def __init__(self, get_convert: Callable[[_HasDataT, StoredT], VisibleT]) -> None:
        self.get_convert = get_convert
        self.name: str = "<not set>"

    def __set_name__(self, owner: type, name: str) -> None:
        if self.name == "<not set>":
            self.name = name

    def __get__(self, instance: _HasDataT, owner=None) -> VisibleT:
        return self.get_convert(instance, cast(StoredT, instance.data[self.name]))


class StaticMethodField(Generic[StoredT, VisibleT]):
    __slots__ = ("set_convert", "get_convert", "name")

    def __init__(self, get_convert: Callable[[StoredT], VisibleT]) -> None:
        self.get_convert = get_convert
        self.name: str = "<not set>"

        def set_convert(value: VisibleT) -> StoredT:
            raise NotImplementedError

        self.set_convert: Callable[[VisibleT], StoredT] = set_convert

    def __set_name__(self, owner: type, name: str) -> None:
        if self.name == "<not set>":
            self.name = name

    def __get__(self, instance: _HasDataT, owner=None) -> VisibleT:
        return self.get_convert(cast(StoredT, instance.data[self.name]))

    def __set__(self, instance: _HasDataT, value: VisibleT) -> None:
        instance.data[self.name] = self.set_convert(value)

    def setter(self, fn: Callable[[VisibleT], StoredT]) -> None:
        self.set_convert = fn


def read_only_field(
    fn: Callable[[_HasDataT, StoredT], VisibleT],
) -> ReadOnlyMethodField[_HasDataT, StoredT, VisibleT]:
    """Decorate a method as getter for a read only field.

    This works similar to the `property` decorator, but will
    automatically pass the content of the `.data` dictionary
    to the decorated method, that matches its name, e.g.,
    decorating a method `name` will call it with `self.data['name']`.
    The method will also be bound as an instance method, i.e.,
    the owning instance will be passed as the first argument.

    Additionally, the name of the decorated method will be registered
    as a required field for all other purposes.

    :::{admonition} Example
    ```python
    from elasticai.creator.ir import IrData, required_field

    class MyData(IrData):
        def __init__(self, data: dict[str, Attribute]):
            self.data = data
            self.length_scaling = 2

        @required_field
        def name(self, value: str) -> str:
            return value

        @type.setter
        def _(self, value: str) -> str:
            return value.lower()
    ```
    :::
    For more information see [`required_field`](#elasticai.creator.ir.required_field).
    """
    return ReadOnlyMethodField(fn)


def static_required_field(
    fn: Callable[[StoredT], VisibleT],
) -> StaticMethodField[StoredT, VisibleT]:
    """Decorate a static method as getter for a read only field.

    Opposed to `read_only_field` this decorator will not pass the owning instance
    to the decorated method. This is to avoid hard to catch inconsistencies
    if the owning instances state changes between read and write operations.

    The main purpose of this decorator is to providea a more readable alternative
    to [`RequiredField`](#elasticai.creator.ir.required_field.RequiredField).

    :::{note}
    Type checkers will not be able to pickup that the decorated methods are static by means of our
    custom descriptor. That means you will probably have to decorate the method with `@staticmethod`
    before applying `@static_required_field`.
    :::

    :::{admonition} Example
    ```python
    from elasticai.creator.ir import IrData, static_required_field

    class MyData(IrData):
        @static_required_field
        @staticmethod
        def length(value: int) -> float:
            return value * 2.0

        @length.setter
        @staticmethod
        def _(value: float) -> int:
            return int(value / 2.0)
    :::

    See also [`read_only_field`](#elasticai.creator.ir.required_field.read_only_field).
    """
    return StaticMethodField(fn)


_required_fields = [
    RequiredField,
    ReadOnlyField,
    SimpleRequiredField,
    StaticMethodField,
    ReadOnlyMethodField,
]


def is_required_field(
    o: object,
) -> TypeIs[RequiredField | ReadOnlyField | SimpleRequiredField]:
    def _isinstance(t):
        return isinstance(o, t)

    return any(map(_isinstance, _required_fields))


def is_required_field_type(
    cls: type,
) -> TypeIs[type[RequiredField] | type[ReadOnlyField] | type[RequiredField]]:
    def _issubclass(t):
        return isinstance(cls, t)

    return any(map(_issubclass, _required_fields))


def register_required_field_type(
    cls: type,
) -> None:
    if cls not in _required_fields:
        _required_fields.append(cls)
