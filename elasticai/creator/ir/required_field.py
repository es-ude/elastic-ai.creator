from collections.abc import Callable
from typing import cast, TypeVar, Generic
from typing_extensions import TypeIs
from elasticai.creator.ir.attribute import Attribute

from ._has_data import HasData


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

    >>> class A:
    ...    number: MandatoryField[str, int] = MandatoryField(set_transform=str, get_transform=int)
    >>> a = A({'number': "12"})
    >>> a.number = a.number + 3
    >>> a.data
    {'number': "15"}
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


class SimpleRequiredField(RequiredField[StoredT, StoredT]):
    slots = ("get_convert", "set_convert", "name")

    def __init__(self):
        super().__init__(lambda x: x, lambda x: x)


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


def is_required_field(o: object) -> TypeIs[RequiredField | ReadOnlyField]:
    return isinstance(o, RequiredField) or isinstance(o, ReadOnlyField)
