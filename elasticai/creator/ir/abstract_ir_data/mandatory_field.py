from collections.abc import Callable
from typing import cast

from typing_extensions import Generic, Protocol, TypeVar

from elasticai.creator.ir.attribute import AttributeT

from ._has_data import HasData

T = TypeVar("T", bound=AttributeT)  # stored data type

F = TypeVar("F", default=T)  # visible data type


class AbstractIR(Protocol):
    data: dict[str, AttributeT]


class TransformableMandatoryField(Generic[T, F]):
    """
    A __descriptor__ that designates a mandatory field of an abstract ir data class.
    The descriptor accesses the `data` dictionary of the owning abstract ir data object
    to read and write values. You can use the `set_transform` and `get_transform` functions
    to transform values during read/write accesses. `T` designates the type stored in the
    `data` dictionary, while `F` is the type that the mandatory field receives.
    That allows to keep dictionary of primitive (serializable) data types in memory,
    while still providing abstract ways to manipulate that data in complex ways.
    This is typically required when working with Nodes and Graphs to create new
    intermediate representations and transform one graph into another.

    E.g.

    ```python
    class A(AbstractIrData):
        number: TransformableMandatoryField[str, int] = TransformableMandatoryField(set_transform=str, get_transform=int)


    a = A({'number': "12"})
    a.number = a.number + 3
    print(a.data) # {'number': "15"}
    ```
    """

    def __init__(
        self,
        set_transform: Callable[[F], T],
        get_transform: Callable[[T], F],
    ):
        self.set_transform = set_transform
        self.get_transform = get_transform

    def __set_name__(self, owner, name: str) -> None:
        """
        IMPORTANT: do not remove owner even though it's not used
        see https://docs.python.org/3/reference/datamodel.html#descriptors for more information
        """
        self.name = name

    def __get__(self, instance: HasData, owner) -> F:
        """
        IMPORTANT: do not remove owner even though it's not used
        see https://docs.python.org/3/reference/datamodel.html#descriptors for more information
        """
        return self.get_transform(cast(T, instance.data[self.name]))

    def __set__(self, instance: HasData, value: F) -> None:
        instance.data[self.name] = self.set_transform(value)


class MandatoryField(TransformableMandatoryField[T, T]):
    def __init__(self):
        super().__init__(lambda x: x, lambda x: x)
