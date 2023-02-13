from abc import abstractmethod
from typing import Iterable, Protocol, Reversible, TypeVar, runtime_checkable

from elasticai.creator.vhdl.typing import Identifiable

T_Node_co = TypeVar("T_Node_co", bound="Node", covariant=True)


class Graph(Protocol[T_Node_co]):
    @property
    @abstractmethod
    def nodes(self) -> Reversible[T_Node_co]:
        ...


@runtime_checkable
class Node(Identifiable, Protocol):
    @property
    @abstractmethod
    def children(self: T_Node_co) -> Iterable[T_Node_co]:
        ...

    @property
    @abstractmethod
    def parents(self: T_Node_co) -> Iterable[T_Node_co]:
        ...


T_contra = TypeVar("T_contra", contravariant=True)


class Sink(Protocol[T_contra]):
    @abstractmethod
    def accepts(self, source: T_contra) -> bool:
        ...
