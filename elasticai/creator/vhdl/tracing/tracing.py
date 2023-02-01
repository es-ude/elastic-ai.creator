import typing
from abc import abstractmethod
from typing import (
    Collection,
    Iterable,
    Optional,
    Protocol,
    Reversible,
    overload,
    runtime_checkable,
)

from torch.fx import Graph as fxGraph

from elasticai.creator.mlframework import Module
from elasticai.creator.vhdl.code import Code, Translatable
from elasticai.creator.vhdl.data_path_connection.typing import Graph
from elasticai.creator.vhdl.data_path_connection.typing import Node as _Node
from elasticai.creator.vhdl.hw_equivalent_layers.typing import HWEquivalentLayer


@typing.runtime_checkable
class TranslatableLayer(Translatable, Module, typing.Protocol):
    @abstractmethod
    def signal_definitions(self, prefix: str) -> Code:
        ...

    @abstractmethod
    def instantiation(self, prefix: str) -> Code:
        ...


@runtime_checkable
class Node(_Node, Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def op(self) -> str:
        ...


@runtime_checkable
class HWEquivalentNode(Node, Protocol):
    @property
    @abstractmethod
    def hw_equivalent_layer(self) -> HWEquivalentLayer:
        ...


class HWEquivalentGraph(Graph[Node], Protocol):
    @overload
    def get_module_for_node(self, node: str) -> Optional[Module]:
        ...

    @overload
    def get_module_for_node(self, node: Node) -> Optional[Module]:
        ...

    @abstractmethod
    def get_module_for_node(self, node: Node | str) -> Optional[Module]:
        ...

    @overload
    def node_has_module(self, node: str) -> bool:
        ...

    @overload
    def node_has_module(self, node: Node) -> bool:
        ...

    @abstractmethod
    def node_has_module(self, node: str | Node) -> bool:
        ...

    @property
    @abstractmethod
    def hw_equivalent_nodes(self) -> Iterable[Node]:
        ...


class HWEquivalentTracer(Protocol):
    @abstractmethod
    def trace(self, root: Module, **kwargs) -> HWEquivalentGraph:
        ...


class _HWEquivalentGraph(HWEquivalentGraph):
    """
        The HWEquivalentGraph is the result of tracing a compatible neural network `m`
    with the corresponding HWEquivalentTracer. It combines signal and
    portmaps for instantiation for all nodes linked to HWEquivalent submodules of `m`
    by making calls to these submodules.
    """

    def __init__(
        self, fx_graph: fxGraph, modules_by_nodes: dict[str, TranslatableLayer]
    ):
        self._fx_graph = fx_graph
        self._modules_by_nodes = modules_by_nodes

    @property
    def nodes(self) -> Reversible[Node]:
        return self._fx_graph.nodes

    def node_has_module(self, node: str | Node) -> bool:
        if isinstance(node, str):
            return node in self._modules_by_nodes
        return node.name in self._modules_by_nodes

    def get_module_for_node(self, node: str | Node) -> Module:
        if isinstance(node, str):
            return self._modules_by_nodes[node]
        return self._modules_by_nodes[node.name]

    @property
    def hw_equivalent_nodes(self) -> Iterable[Node]:
        return filter(lambda n: n.name in self._modules_by_nodes, self._fx_graph.nodes)

    @property
    def hw_equivalent_layers(self) -> Collection[HWEquivalentLayer]:
        layers = set()
        for node in self.hw_equivalent_nodes:
            if isinstance(node, HWEquivalentNode):
                layers.add(node.hw_equivalent_layer)
        return layers
