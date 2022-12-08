from typing import Protocol, Iterable, Optional, Any, Callable

import torch
from torch.fx import Tracer as fxTracer
from torch.fx import Graph as fxGraph
from torch.fx import Node as fxNode
from torch.fx.node import Target, Argument

from elasticai.creator.mlframework import Module


class Node(Protocol):
    @property
    def module(self) -> Module:
        ...


class Graph(Protocol):
    @property
    def nodes(self) -> Iterable[Node]:
        ...


class Tracer(fxTracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return m.__module__.startswith("elasticai.creator.vhdl.hw_equivalent_layers")

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
