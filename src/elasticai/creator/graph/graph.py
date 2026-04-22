from abc import abstractmethod
from collections.abc import Collection, Hashable, Iterator, Mapping
from typing import Protocol, Self


class ReadOnlyGraph[T: Hashable](Protocol):
    @property
    @abstractmethod
    def nodes(self) -> Collection[T]: ...

    @abstractmethod
    def iter_edges(self) -> Iterator[tuple[T, T]]: ...

    @property
    @abstractmethod
    def predecessors(self) -> Mapping[T, Collection[T]]: ...

    @property
    @abstractmethod
    def successors(self) -> Mapping[T, Collection[T]]: ...


class Graph[T: Hashable](ReadOnlyGraph[T], Protocol):
    @abstractmethod
    def add_node(self, node: T) -> Self: ...

    @abstractmethod
    def add_edge(self, src: T, dst: T) -> Self: ...

    @abstractmethod
    def new(self) -> "Graph[T]": ...

    @abstractmethod
    def copy(self) -> "Graph[T]": ...
