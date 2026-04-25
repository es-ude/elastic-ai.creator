from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol

from ._attribute import AttributeMapping
from .datagraph import DataGraph, Graph, NodeEdgeFactory, ReadOnlyDataGraph
from .datagraph import Edge as _Edge
from .datagraph import Node as _Node
from .graph import GraphImpl
from .registry import Registry


class StdNodeEdgeFactory[N: _Node, E: _Edge](NodeEdgeFactory):
    def __init__(
        self,
        node_fn: Callable[[str, AttributeMapping], N],
        edge_fn: Callable[[str, str, AttributeMapping], E],
    ) -> None:
        self._node_fn = node_fn
        self._edge_fn = edge_fn

    def node(self, name: str, attributes: AttributeMapping = AttributeMapping()) -> N:
        return self._node_fn(name, attributes)

    def edge(
        self, src: str, dst: str, attributes: AttributeMapping = AttributeMapping()
    ) -> E:
        return self._edge_fn(src, dst, attributes)


class IrFactory[N: _Node, E: _Edge, G: DataGraph](NodeEdgeFactory[N, E], Protocol):
    @abstractmethod
    def node_from_other(self, other: _Node) -> N:
        return self.node(other.name, other.attributes)

    @abstractmethod
    def edge_from_other(self, other: _Edge) -> E:
        return self.edge(other.src, other.dst, other.attributes)

    @abstractmethod
    def graph(
        self,
        attributes: AttributeMapping = AttributeMapping(),
    ) -> G: ...

    @abstractmethod
    def graph_from_other(self, other: ReadOnlyDataGraph) -> G: ...

    @abstractmethod
    def registry(
        self,
        items: Mapping[str, DataGraph] | Iterable[tuple[str, DataGraph]] | None = None,
    ) -> Registry[G]: ...


class _GraphConstructorMethod[G: DataGraph](Protocol):
    def __call__(
        self,
        /,
        *,
        factory: NodeEdgeFactory,
        attributes: AttributeMapping,
        graph: Graph[str, AttributeMapping],
        node_attributes: AttributeMapping,
    ) -> G: ...


class _GraphConstructorFn[G: DataGraph](Protocol):
    @staticmethod
    def __call__(
        *,
        factory: NodeEdgeFactory,
        attributes: AttributeMapping,
        graph: Graph[str, AttributeMapping],
        node_attributes: AttributeMapping,
    ) -> G: ...


type _GraphConstructor[G: DataGraph] = (
    _GraphConstructorFn[G] | _GraphConstructorMethod[G]
)


class StdIrFactory[N: _Node, E: _Edge, G: DataGraph](IrFactory[N, E, G]):
    def __init__(
        self,
        node_fn: Callable[[str, AttributeMapping], N],
        edge_fn: Callable[[str, str, AttributeMapping], E],
        graph_fn: _GraphConstructor[G],
    ) -> None:
        self._node = node_fn
        self._edge = edge_fn
        self._graph = graph_fn

    def registry(
        self,
        items: Mapping[str, DataGraph] | Iterable[tuple[str, DataGraph]] | None = None,
    ) -> Registry[G]:
        if items is None:
            return Registry()
        reg = Registry(items)
        return reg.apply(lambda g: self.graph_from_other(other=g))

    def node(
        self,
        name: str,
        attributes: AttributeMapping = AttributeMapping(),
    ) -> N:
        return self._node(name, attributes)

    def node_from_other(self, other: _Node) -> N:
        return self._node(other.name, other.attributes)

    def edge(
        self,
        src: str,
        dst: str,
        attributes: AttributeMapping = AttributeMapping(),
    ) -> E:
        return self._edge(src, dst, attributes)

    def edge_from_other(self, other: _Edge) -> E:
        return self._edge(other.src, other.dst, other.attributes)

    def graph(
        self,
        attributes: AttributeMapping = AttributeMapping(),
    ) -> G:
        """create a new graph using the underlying data from other"""
        empty_attributes = AttributeMapping()
        return self._graph(
            factory=self,
            attributes=attributes,
            graph=GraphImpl(lambda: empty_attributes),
            node_attributes=empty_attributes,
        )

    def graph_from_other(self, other: ReadOnlyDataGraph) -> G:
        return self._graph(
            factory=self,
            attributes=other.attributes,
            graph=other.graph,
            node_attributes=other.node_attributes,
        )
