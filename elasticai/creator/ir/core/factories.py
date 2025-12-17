from abc import abstractmethod
from typing import Protocol

from .attribute import AttributeMapping
from .graph import Graph


class _Node(Protocol):
    @property
    def attributes(self) -> AttributeMapping: ...

    @property
    def name(self) -> str: ...


class _Edge(Protocol):
    @property
    def attributes(self) -> AttributeMapping: ...

    @property
    def src(self) -> str: ...

    @property
    def dst(self) -> str: ...


class DataGraph[N: _Node, E: _Edge](Graph[str, AttributeMapping], Protocol):
    def with_attributes(self, attributes: AttributeMapping) -> "DataGraph[N, E]": ...


class NodeEdgeFactory[N: _Node, E: _Edge](Protocol):
    @abstractmethod
    def node(self, name: str, attributes: AttributeMapping) -> N: ...

    @abstractmethod
    def edge(self, src: str, dst: str, attributes: AttributeMapping) -> E: ...


class IrFactory[N: _Node, E: _Edge, G: DataGraph](NodeEdgeFactory[N, E], Protocol):
    @abstractmethod
    def graph(self, attributes: AttributeMapping) -> G: ...
