import copy
from collections.abc import Sequence
from typing import TypeVar, cast

from elasticai.creator import graph as g
from elasticai.creator import ir
from elasticai.creator.torch2ir import Implementation

from ._matcher import Matcher as _Matcher
from .constraint import NodeConstraint


def build_sequential_pattern(
    sequence: Sequence[ir.Node],
) -> ir.Implementation[ir.Node, ir.Edge]:
    pattern = ir.Implementation(graph=g.BaseGraph())
    previous = None
    for node in sequence:
        pattern.add_node(node)
        if previous is not None:
            pattern.add_edge(src=previous, dst=node.name)
        previous = node.name

    return pattern


PNode = TypeVar("PNode", bound=ir.Node)
GNode = TypeVar("GNode", bound=ir.Node)


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
        self.interface = g.BaseGraph().add_node("start").add_node("end")
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
        rewriter = g.GraphRewriter(
            pattern=self.pattern.graph,
            interface=self.interface,
            replacement=self.replacement.graph,
            match=self.matcher,
            lhs=self.lhs,
            rhs=self.rhs,
        )
        result = rewriter.rewrite(impl.graph)
        self._has_changed = len(result.pattern_to_original) == 0

        new_data = copy.deepcopy(impl.data)

        def copy_original_data_to_replaced_subgraph(
            pattern_node: str, replacement_node: str
        ) -> None:
            new_name = result.replacement_to_new[replacement_node]

            cast(dict[str, ir.Attribute], new_data["nodes"])[new_name] = cast(
                dict,
                copy.deepcopy(
                    cast(dict[str, ir.Attribute], new_data["nodes"])[
                        cast(str, result.pattern_to_original[pattern_node])
                    ]
                ),
            ) | {"name": new_name}

        for node in self._get_nodes_that_are_in_pattern_and_replacement():
            copy_original_data_to_replaced_subgraph(node, node)
        return ir.Implementation(graph=result.new_graph, data=new_data)
