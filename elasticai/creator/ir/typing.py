from typing import Protocol, Self, TypeVar, runtime_checkable

from .attribute import AttributeT


@runtime_checkable
class Node(Protocol):
    name: str
    type: str
    data: dict[str, AttributeT]
    attributes: dict[str, AttributeT]

    @classmethod
    def new(cls, *args, **kwargs) -> Self: ...

    @classmethod
    def from_dict(cls, d: dict[str, AttributeT]) -> Self: ...

    def as_dict(self) -> dict[str, AttributeT]: ...


NodeT = TypeVar("NodeT", bound=Node)
