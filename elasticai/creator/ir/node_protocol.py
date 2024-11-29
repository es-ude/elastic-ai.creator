from typing import Protocol, Self, TypeVar, runtime_checkable

from .attribute import Attribute


@runtime_checkable
class Node(Protocol):
    name: str
    type: str
    data: dict[str, Attribute]
    attributes: dict[str, Attribute]

    @classmethod
    def new(cls, *args, **kwargs) -> Self: ...

    @classmethod
    def from_dict(cls, d: dict[str, Attribute]) -> Self: ...

    def as_dict(self) -> dict[str, Attribute]: ...


NodeT = TypeVar("NodeT", bound=Node)
