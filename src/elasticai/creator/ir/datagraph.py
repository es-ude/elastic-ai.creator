from abc import abstractmethod
from collections.abc import Mapping
from typing import Protocol, Self, TypeVar, overload, runtime_checkable

from ._attribute import AttributeMapping
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
    def __eq__(self, o: object, /) -> bool: ...


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

    @abstractmethod
    def __eq__(self, o: object, /) -> bool: ...


N = TypeVar("N", bound=Node, covariant=True)
E = TypeVar("E", bound=Edge, covariant=True)


class NodeEdgeFactory(Protocol[N, E]):
    @abstractmethod
    def node(
        self, name: str, attributes: AttributeMapping = AttributeMapping(), /
    ) -> N: ...

    @abstractmethod
    def edge(
        self, src: str, dst: str, attributes: AttributeMapping = AttributeMapping(), /
    ) -> E: ...


@runtime_checkable
class ReadOnlyDataGraph(ReadOnlyGraph[str, AttributeMapping], Protocol[N, E]):
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


class DataGraph(ReadOnlyDataGraph[N, E], Graph[str, AttributeMapping], Protocol[N, E]):
    """
    Note that the fact, that `add_node` and similar methods take
    an argument of type `Node` while the `nodes` and `edges` properties
    produce mappings over the generic type parameters `E` and `N`.
    This makes `DataGraph` covariant over `N` and `E`, meaning
    that `DataGraph[SpecialNode, SpecialEdge]` is a subtype
    of `DataGraph[Node, Edge]`. It also means that every implementation
    detail in a `DataGraph` implementation needs to assume that
    nodes and edges are of the most general `Node` or `Edge` type.
    """

    @overload
    def add_node(self, node: Node, /) -> Self: ...

    @overload
    def add_node(
        self, name: str, attributes: AttributeMapping | None = None, /
    ) -> Self: ...

    @abstractmethod
    def add_nodes(self, *args: Node | tuple[str, AttributeMapping] | str) -> Self:
        """Updates node in case it exists already."""
        ...

    @overload
    def add_edge(self, edge: Edge, /) -> Self: ...

    @overload
    def add_edge(
        self, src: str, dst: str, attributes: AttributeMapping | None = None, /
    ) -> Self: ...

    @abstractmethod
    def add_edges(
        self, *args: Edge | tuple[str, str, AttributeMapping] | tuple[str, str]
    ) -> Self:
        """Updates edge in case it exists already. Possibly already existing nodes remain unchanged."""
        ...

    @abstractmethod
    def remove_node(self, node: str, /) -> Self:
        """Will remove node and all connected edges."""
        ...

    @abstractmethod
    def remove_edge(self, src: str, dst: str) -> Self:
        """Will not remove nodes, even if they become isolated."""
        ...

    @abstractmethod
    def with_attributes(self, attributes: AttributeMapping) -> Self: ...

    @abstractmethod
    def clear(self) -> Self:
        """Get a new empty graph with attributes, nodes, edges removed."""
        ...
