from abc import abstractmethod
from typing import Protocol, Iterable, Any, Callable

import torch
from torch.fx import Tracer as fxTracer
from torch.fx import Graph as fxGraph
from elasticai.creator.mlframework import Module


class Node(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...


class ModuleNode(Node, Protocol):
    @property
    @abstractmethod
    def module(self) -> Module:
        ...


class Graph(Protocol):
    @property
    @abstractmethod
    def module_nodes(self) -> Iterable[ModuleNode]:
        ...

    @property
    @abstractmethod
    def nodes(self) -> Iterable[Node]:
        ...


class Tracer(Protocol):
    @abstractmethod
    def trace(self, model: Module) -> Graph:
        ...


class HWEquivalentTracer(Tracer):
    def __init__(self):
        self._tracer = _HWEquivalentTracer()

    def trace(self, model: Module) -> Graph:
        return self._tracer.trace(
            model,
        )


class _Graph(Graph):
    def __init__(self, fx_graph: fxGraph):
        self._fx_graph = fx_graph

    def __getattr__(self, item):
        return getattr(self._fx_graph, item)

    @property
    def nodes(self):
        return self._fx_graph.nodes

    @property
    def module_nodes(self) -> Iterable[ModuleNode]:
        yield from filter(lambda n: n.op == "call_module", self._fx_graph.nodes)


class _HWEquivalentTracer(fxTracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return m.__module__.startswith("elasticai.creator.vhdl.hw_equivalent_layers")

    def trace(self, root, **kwargs) -> Graph:
        graph = super().trace(root, **kwargs)
        return _Graph(graph)

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        proxy = super().call_module(m, forward, args, kwargs)
        proxy.node.module = m
        return proxy
