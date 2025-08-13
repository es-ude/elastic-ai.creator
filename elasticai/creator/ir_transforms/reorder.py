import copy
from collections.abc import Sequence
from typing import Protocol, TypeVar, cast

from elasticai.creator import graph as g
from elasticai.creator import ir
from elasticai.creator.graph import Graph
from elasticai.creator.torch2ir import Implementation


def build_sequential_pattern(
    sequence: Sequence[ir.Node],
) -> ir.Implementation[ir.Node, ir.Edge]:
    pattern = ir.Implementation(graph=g.BaseGraph())
    previous = None
    for node in sequence:
        pattern.add_node(node)
        if previous is not None:
            pattern.add_edge(src=previous, dst=node.name, data={})
        previous = node.name

    return pattern


PNode = TypeVar("PNode", bound=ir.Node)
GNode = TypeVar("GNode", bound=ir.Node)

PNodeCon = TypeVar("PNodeCon", bound=ir.Node, contravariant=True)
GNodeCon = TypeVar("GNodeCon", bound=ir.Node, contravariant=True)


class NodeConstraint(Protocol[PNodeCon, GNodeCon]):
    def __call__(self, *, pattern_node: PNodeCon, graph_node: GNodeCon) -> bool: ...


class _Matcher:
    def __init__(
        self,
        pattern: ir.Implementation[ir.Node, ir.Edge],
        graph: ir.Implementation[ir.Node, ir.Edge],
        node_constraint: NodeConstraint[ir.Node, ir.Node],
    ):
        self.pattern = pattern
        self.graph = graph
        self._node_constraint = node_constraint

    def set_graph(self, graph: ir.Implementation[ir.Node, ir.Edge]) -> None:
        self.graph = graph

    def node_constraint(self, pattern_node: str, graph_node: str) -> bool:
        return self._node_constraint(
            pattern_node=self.pattern.nodes[pattern_node],
            graph_node=self.graph.nodes[graph_node],
        )

    def __call__(self, pattern: Graph[str], graph: Graph[str]) -> dict[str, str]:
        return g.find_subgraph(
            pattern=pattern, graph=graph, node_constraint=self.node_constraint
        )


class SequenceReorderer:
    def __init__(
        self,
        old_order: Sequence[ir.Node],
        new_order: Sequence[ir.Node],
        node_constraint: NodeConstraint[ir.Node, ir.Node],
    ):
        """Use this to reorder the first occurence of a pattern as specified by the replacement."""
        pattern = old_order
        replacement = new_order
        self.pattern = build_sequential_pattern(pattern)
        self.replacement = build_sequential_pattern(replacement)
        self.lhs = {"start": pattern[0].name, "end": pattern[-1].name}
        self.rhs = {"start": replacement[0].name, "end": replacement[-1].name}
        self.matcher = _Matcher(
            pattern=self.pattern,
            graph=Implementation(data={}, graph=g.BaseGraph()),
            node_constraint=node_constraint,
        )
        self._has_changed = False

    def _get_nodes_that_are_in_pattern_and_replacement(self) -> set[str]:
        return set(self.pattern.nodes) & set(self.replacement.nodes)

    def has_changed(self) -> bool:
        """Tells whether the graph has been changed by the last call to reorder."""
        return self._has_changed

    def reorder(self, impl: ir.Implementation) -> ir.Implementation:
        """Reorder the first occurence of the pattern as specified by the replacement.

        The resulting graph and its data are deep copies of the input graph and its data.
        The new node names are unique but chosen based on the replacement nodes and
        not based on the original node names.
        Use `has_changed` to check whether the graph has been changed by the last call to `reorder`
        if you want to keep reordering until you reach a fixed point.

        :::{note}
        Will create a deep copy even if no changes were made.
        As such it is always safe to `del original_graph` after calling `reorder`.
        :::
        """
        self.matcher.set_graph(impl)
        match = self.matcher(self.pattern.graph, impl.graph)
        new_graph, new_names = g.rewrite(
            original=impl.graph,
            replacement=self.replacement.graph,
            match=match,
            lhs=self.lhs,
            rhs=self.rhs,
        )
        self._has_changed = len(match) > 0

        new_data = copy.deepcopy(impl.data)

        def copy_original_data_to_replaced_subgraph(
            pattern_node: str, replacement_node: str
        ) -> None:
            new_name = new_names[replacement_node]

            cast(dict[str, ir.Attribute], new_data["nodes"])[new_name] = cast(
                dict,
                copy.deepcopy(
                    cast(dict[str, ir.Attribute], new_data["nodes"])[
                        cast(str, match[pattern_node])
                    ]
                ),
            )

        for node in self._get_nodes_that_are_in_pattern_and_replacement():
            copy_original_data_to_replaced_subgraph(node, node)
        return ir.Implementation(graph=new_graph, data=new_data)
