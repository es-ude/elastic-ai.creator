from collections.abc import Mapping
from typing import Any

from .rewriter import (
    CompositeRule,
    DataGraph,
    IrFactory,
    Pattern,
    PatternRuleSpec,
    ReadOnlyDataGraph,
    Rule,
    StdPattern,
)
from .rewriter import (
    PatternRule as _PatternRule,
)

__all__ = [
    "Rewriter",
    "PatternRuleSpec",
    "ReadOnlyDataGraph",
    "IrFactory",
    "Rule",
    "DataGraph",
    "CompositeRule",
    "PatternRule",
    "Pattern",
    "StdPattern",
]

from elasticai.creator.graph import BaseGraph as _BaseGraph

from ..core import Edge as _Edge
from ..core import Implementation as _Implementation
from ..core import Node as _Node


class _DefaultFactory(IrFactory):
    def node(self, name: str, data: Mapping[str, Any]) -> _Node:
        return _Node(name=name, data=dict(data))

    def edge(self, src: str, dst: str, data: Mapping[str, Any]) -> _Edge:
        return _Edge(src=src, dst=dst, data=dict(data))

    def data_graph(self) -> DataGraph[_Node, _Edge]:
        return _Implementation(graph=_BaseGraph(), data={})


class PatternRule(_PatternRule):
    def __init__(self, spec: PatternRuleSpec):
        super().__init__(ir_factory=_DefaultFactory(), spec=spec)
