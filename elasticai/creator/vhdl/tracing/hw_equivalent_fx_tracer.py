from typing import Any, Callable, cast

import torch
from torch.fx import Tracer as fxTracer

from elasticai.creator.mlframework import Module
from elasticai.creator.vhdl.tracing.hw_equivalent_graph_impl import _HWEquivalentGraph
from elasticai.creator.vhdl.tracing.typing import (
    HWEquivalentGraph,
    HWEquivalentTracer,
    TranslatableLayer,
)


class HWEquivalentFXTracer(fxTracer, HWEquivalentTracer[TranslatableLayer]):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return m.__module__.startswith("elasticai.creator.vhdl.hw_equivalent")

    def __init__(self) -> None:
        super().__init__()
        self._modules_by_nodes: dict[str, TranslatableLayer] = dict()

    def trace(self, root: Module, **kwargs) -> HWEquivalentGraph:
        graph = super().trace(root, **kwargs)
        return _HWEquivalentGraph(graph, self._modules_by_nodes)

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        proxy = super().call_module(m, forward, args, kwargs)
        self._modules_by_nodes[proxy.node.name] = cast(TranslatableLayer, m)
        return proxy
