from abc import abstractmethod
from itertools import chain
from typing import Any, Callable, Collection, Iterable, Protocol, runtime_checkable

import torch
from torch.fx import Graph as fxGraph
from torch.fx import Tracer as fxTracer

from elasticai.creator.mlframework import Module
from elasticai.creator.vhdl.code import Code
from elasticai.creator.vhdl.hw_equivalent_layers.typing import HWEquivalentLayer


class Node(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def next(self) -> "Node":
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


class HWBlockCollection(Protocol):
    @abstractmethod
    def signals(self, prefix: str) -> Code:
        ...

    @abstractmethod
    def instantiations(self, prefix: str) -> Code:
        ...


class HWEquivalentGraph(Protocol):
    @property
    @abstractmethod
    def hw_equivalent_nodes(self) -> Iterable[HWEquivalentNode]:
        ...

    @property
    @abstractmethod
    def nodes(self) -> Iterable[Node]:
        ...


class Tracer(Protocol):
    @abstractmethod
    def trace(self, model: Module) -> HWEquivalentGraph:
        ...


class HWEquivalentTracer(Tracer):
    def __init__(self):
        self._tracer = _HWEquivalentTracer()

    def trace(self, model: Module) -> HWEquivalentGraph:
        return self._tracer.trace(
            model,
        )


def create_hw_block_collection(graph: HWEquivalentGraph) -> HWBlockCollection:
    if not isinstance(graph, _HWEquivalentGraph):
        raise NotImplementedError(
            "hw block creation unsupported for {}".format(type(graph))
        )
    else:
        return graph


class _HWEquivalentGraph(HWEquivalentGraph, HWBlockCollection):
    """
        The HWEquivalentGraph is the result of tracing a compatible neural network `m`
    with the corresponding HWEquivalentTracer. It combines signal and
    portmaps for instantiation for all nodes linked to HWEquivalent submodules of `m`
    by making calls to these submodules.
    """

    def signals(self, prefix: str) -> Code:
        yield from chain.from_iterable(
            (
                node.hw_equivalent_layer.signal_definitions(f"{prefix}{node.name}")
                for node in self.hw_equivalent_nodes
            )
        )

    def _call_on_layers(self, method_name, prefix: str) -> Code:
        yield from chain.from_iterable(
            (
                getattr(node.hw_equivalent_layer, method_name)(f"{prefix}{node.name}")
                for node in self.hw_equivalent_nodes
            )
        )

    def instantiations(self, prefix: str) -> Code:
        yield from chain.from_iterable(
            (
                node.hw_equivalent_layer.instantiation(f"{prefix}{node.name}")
                for node in self.hw_equivalent_nodes
            )
        )

    def __init__(self, fx_graph: fxGraph):
        self._fx_graph = fx_graph

    @property
    def nodes(self):
        yield from self._fx_graph.nodes

    @property
    def hw_equivalent_nodes(self) -> Iterable[HWEquivalentNode]:
        yield from filter(lambda n: n.op == "call_module", self._fx_graph.nodes)

    @property
    def hw_equivalent_layers(self) -> Collection[HWEquivalentLayer]:
        layers = set()
        for node in self.hw_equivalent_nodes:
            if isinstance(node, HWEquivalentNode):
                layers.add(node.hw_equivalent_layer)
        return layers


class _HWEquivalentTracer(fxTracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return m.__module__.startswith("elasticai.creator.vhdl.hw_equivalent_layers")

    def trace(self, root, **kwargs) -> HWEquivalentGraph:
        graph = super().trace(root, **kwargs)
        return _HWEquivalentGraph(graph)

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        proxy = super().call_module(m, forward, args, kwargs)
        proxy.node.hw_equivalent_layer = m
        return proxy
