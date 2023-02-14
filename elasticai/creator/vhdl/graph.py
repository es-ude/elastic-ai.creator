"""
The module contains protocol types/abstract classes to
provide a common interface for graph
data structures throughout the library
"""
from abc import abstractmethod
from typing import Iterable, Protocol, Reversible, TypeVar, runtime_checkable

from elasticai.creator.vhdl.typing import Identifiable

N_co = TypeVar("N_co", bound="Node", covariant=True)


class Graph(Protocol[N_co]):
    @property
    @abstractmethod
    def nodes(self) -> Reversible[N_co]:
        ...


@runtime_checkable
class Node(Identifiable, Protocol):
    @property
    @abstractmethod
    def children(self: N_co) -> Iterable[N_co]:
        ...

    @property
    @abstractmethod
    def parents(self: N_co) -> Iterable[N_co]:
        ...
