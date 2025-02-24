from typing import Generic

import elasticai.creator.graph as g
from elasticai.creator import ir
from elasticai.creator.graph import Graph

from ._types import GNode, PNode
from .constraint import NodeConstraint


class Matcher(Generic[PNode, GNode]):
    def __init__(
        self,
        pattern: ir.Implementation[PNode, ir.Edge],
        graph: ir.Implementation[GNode, ir.Edge],
        node_constraint: NodeConstraint[PNode, GNode],
    ):
        self.pattern = pattern
        self.graph = graph
        self._node_constraint = node_constraint

    def set_graph(self, graph: ir.Implementation[GNode, ir.Edge]) -> None:
        self.graph = graph

    def node_constraint(self, pattern_node: str, graph_node: str) -> bool:
        return self._node_constraint(
            pattern_node=self.pattern.nodes[pattern_node],
            graph_node=self.graph.nodes[graph_node],
        )

    def __call__(self, pattern: Graph[str], graph: Graph[str]) -> list[dict[str, str]]:
        return g.find_subgraphs(
            pattern=pattern, graph=graph, node_constraint=self.node_constraint
        )
