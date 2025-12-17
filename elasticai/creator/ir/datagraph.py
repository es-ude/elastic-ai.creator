from abc import abstractmethod
from collections.abc import Mapping
from typing import Protocol, Self, overload, runtime_checkable

from .attribute import AttributeMapping
from .graph import Graph, ReadOnlyGraph


@runtime_checkable
class Node(Protocol):
    @property
    @abstractmethod
    def attributes(self) -> AttributeMapping: ...

    @property
    @abstractmethod
    def type(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __eq__(self, o: object) -> bool: ...


@runtime_checkable
class Edge(Protocol):
    @property
    @abstractmethod
    def attributes(self) -> AttributeMapping: ...

    @property
    @abstractmethod
    def src(self) -> str: ...

    @property
    @abstractmethod
    def dst(self) -> str: ...


@runtime_checkable
class ReadOnlyDataGraph[N: Node, E: Edge](
    ReadOnlyGraph[str, AttributeMapping], Protocol
):
    @property
    @abstractmethod
    def attributes(self) -> AttributeMapping: ...

    @property
    @abstractmethod
    def nodes(self) -> Mapping[str, N]: ...

    @property
    @abstractmethod
    def graph(self) -> Graph[str, AttributeMapping]: ...

    @property
    @abstractmethod
    def node_attributes(self) -> AttributeMapping: ...

    @property
    @abstractmethod
    def edges(self) -> Mapping[tuple[str, str], E]: ...


class DataGraph[N, E](ReadOnlyDataGraph, Graph[str, AttributeMapping], Protocol):
    @overload
    def add_node(self, node: N, /) -> Self: ...

    @overload
    def add_node(
        self, name: str, attributes: AttributeMapping | None = None, /
    ) -> Self: ...

    @abstractmethod
    def add_nodes(self, *args: N | tuple[str, AttributeMapping] | str) -> Self: ...

    @overload
    def add_edge(self, edge: E, /) -> Self: ...

    @overload
    def add_edge(
        self, src: str, dst: str, attributes: AttributeMapping | None = None, /
    ) -> Self: ...

    @abstractmethod
    def add_edges(
        self, *args: E | tuple[str, str, AttributeMapping] | tuple[str, str]
    ) -> Self: ...

    @abstractmethod
    def remove_node(self, node: str, /) -> Self: ...

    @abstractmethod
    def remove_edge(self, src: str, dst: str) -> Self: ...

    @abstractmethod
    def with_attributes(self, attributes: AttributeMapping) -> Self: ...

    @abstractmethod
    def new(self) -> Self: ...
