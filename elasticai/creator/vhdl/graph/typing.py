from abc import abstractmethod
from typing import Iterable, Protocol, Reversible, TypeVar

T_Node_co = TypeVar("T_Node_co", bound="Node", covariant=True)


class Graph(Protocol[T_Node_co]):
    @property
    @abstractmethod
    def nodes(self) -> Reversible[T_Node_co]:
        ...


T_Node = TypeVar("T_Node", bound="Node")
T_co = TypeVar("T_co", covariant=True)


class Node(Protocol[T_co]):
    @property
    @abstractmethod
    def children(self: "Node[T_co]") -> Iterable["Node[T_co]"]:
        ...

    @property
    @abstractmethod
    def parents(self: "Node[T_co]") -> Iterable["Node[T_co]"]:
        ...
