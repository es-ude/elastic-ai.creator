from abc import abstractmethod
from collections.abc import Iterator, Mapping, Set
from typing import Protocol, Self, TypeVar

T = TypeVar("T", str, int)


class Graph(Protocol[T]):
    @property
    @abstractmethod
    def nodes(self) -> Set[T]: ...

    @abstractmethod
    def iter_edges(self) -> Iterator[tuple[T, T]]: ...

    @abstractmethod
    def add_node(self, node: T) -> Self: ...

    @abstractmethod
    def add_edge(self, src: T, sink: T) -> Self: ...

    @property
    @abstractmethod
    def predecessors(self) -> Mapping[T, set[T]]: ...

    @property
    @abstractmethod
    def successors(self) -> Mapping[T, set[T]]: ...

    @abstractmethod
    def new(self) -> "Graph[T]": ...
