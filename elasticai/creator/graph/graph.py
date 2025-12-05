from abc import abstractmethod
from collections.abc import Hashable, Iterator, Mapping, Set
from typing import Protocol, Self


class ReadOnlyGraph[T: Hashable](Protocol):
    @property
    @abstractmethod
    def nodes(self) -> Set[T]: ...

    @abstractmethod
    def iter_edges(self) -> Iterator[tuple[T, T]]: ...

    @property
    @abstractmethod
    def predecessors(self) -> Mapping[T, set[T]]: ...

    @property
    @abstractmethod
    def successors(self) -> Mapping[T, set[T]]: ...


class Graph[T: Hashable](ReadOnlyGraph[T], Protocol):
    @abstractmethod
    def add_node(self, node: T) -> Self: ...

    @abstractmethod
    def add_edge(self, src: T, dst: T) -> Self: ...

    @abstractmethod
    def new(self) -> "Graph[T]": ...

    @abstractmethod
    def copy(self) -> "Graph[T]": ...
