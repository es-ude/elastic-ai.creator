from abc import abstractmethod
from collections.abc import Callable
from typing import Protocol

from .attribute import AttributeMapping
from .datagraph import DataGraph, NodeEdgeFactory
from .datagraph import Edge as _Edge
from .datagraph import Node as _Node


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


class DataGraphFactory[G: DataGraph](Protocol):
    def graph(self, attributes: AttributeMapping = AttributeMapping(), /) -> G: ...


class StdDataGraphFactory[N: _Node, E: _Edge, G: DataGraph](DataGraphFactory[G]):
    def __init__(
        self,
        node_edge: NodeEdgeFactory[N, E],
        graph_fn: Callable[[NodeEdgeFactory[N, E], AttributeMapping], G],
    ):
        self._node_edge = node_edge
        self._graph_fn = graph_fn

    def graph(
        self,
        attributes: AttributeMapping = AttributeMapping(),
    ) -> G:
        return self._graph_fn(
            self._node_edge,
            attributes,
        )


class IrFactory[N: _Node, E: _Edge, G: DataGraph](NodeEdgeFactory[N, E], Protocol):
    @abstractmethod
    def graph(self, attributes: AttributeMapping = AttributeMapping(), /) -> G: ...


class StdIrFactory[N: _Node, E: _Edge, G: DataGraph](StdNodeEdgeFactory[N, E]):
    def __init__(
        self,
        node_fn: Callable[[str, AttributeMapping], N],
        edge_fn: Callable[[str, str, AttributeMapping], E],
        graph_fn: Callable[[NodeEdgeFactory[N, E], AttributeMapping], G],
    ) -> None:
        super().__init__(node_fn, edge_fn)
        self._graph = StdDataGraphFactory(self, graph_fn)

    def graph(self, attributes: AttributeMapping = AttributeMapping()) -> G:
        return self._graph.graph(attributes)
