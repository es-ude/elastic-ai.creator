__all__ = [
    "RequiredField",
    "SimpleRequiredField",
    "IrData",
    "Edge",
    "Node",
    "edge",
    "node",
    "LoweringPass",
    "Lowerable",
    "Implementation",
    "Attribute",
    "static_required_field",
    "read_only_field",
    "StaticMethodField",
    "ReadOnlyField",
    "GraphProtocol",
    "PatternRuleSpec",
    "ReadOnlyDataGraph",
    "Rule",
    "CompositeRule",
    "Pattern",
    "StdPattern",
]
from collections.abc import Mapping
from typing import Any

from elasticai.creator.graph import BaseGraph as _BaseGraph

from .base import (
    Attribute,
    IrData,
    ReadOnlyField,
    RequiredField,
    SimpleRequiredField,
    StaticMethodField,
    read_only_field,
    static_required_field,
)
from .core import (
    Edge,
    Implementation,
    Lowerable,
    LoweringPass,
    Node,
    edge,
    node,
)
from .core import Edge as _Edge
from .core import (
    Graph as GraphProtocol,
)
from .core import Implementation as _Implementation
from .core import Node as _Node
from .rewriting import (
    CompositeRule,
    IrFactory,
    Pattern,
    PatternRuleSpec,
    ReadOnlyDataGraph,
    Rule,
    StdPattern,
)
from .rewriting import DataGraph as _rewritingDataGraph
from .rewriting import (
    PatternRule as _PatternRule,
)


class _DefaultFactory(IrFactory):
    def node(self, name: str, data: Mapping[str, Any]) -> _Node:
        return _Node(name=name, data=dict(data))

    def edge(self, src: str, dst: str, data: Mapping[str, Any]) -> _Edge:
        return _Edge(src=src, dst=dst, data=dict(data))

    def data_graph(self) -> _rewritingDataGraph[_Node, _Edge]:
        return _Implementation(graph=_BaseGraph(), data={})


class PatternRule(_PatternRule):
    def __init__(self, spec: PatternRuleSpec):
        super().__init__(ir_factory=_DefaultFactory(), spec=spec)
