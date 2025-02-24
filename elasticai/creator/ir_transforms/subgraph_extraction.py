from typing import Any, Generic, cast

from elasticai.creator import graph as g
from elasticai.creator import ir

from ._matcher import Matcher
from ._types import GNode, PNode
from .constraint import NodeConstraint


class SubgraphExtractor(Generic[PNode, GNode]):
    def __init__(
        self,
        pattern: ir.Implementation[PNode, ir.Edge],
        node_constraint: NodeConstraint[PNode, GNode],
    ):
        """Extract pattern from a graph, where it matches the node constraint.

        The `"input"` and `"output"` nodes of the pattern will automatically be used
        as the interface (see [`g.GraphRewriter`](#elasticai.creator.graph.GraphRewriter) for more details).

        The call to `.extract()` will return two new implementations:
          1. where the pattern has been replaced by a single node
          2. the extracted subgraph implementation


        Example:
        ```python
        # Define a pattern to match (e.g., two nodes of type 'T' in sequence)
        pattern = (ir.Implementation(data=dict(name="pattern", type="P"))
                    .add_node(name="input", type="any")
                    .add_node(name="output", type="any")
                    .add_node(name="a", type="T")
                    .add_node(name="b", type="T")
                    .add_edge(src="input", dst="a")
                    .add_edge(src="a", dst="b")
                    .add_edge(src="b", dst="output"))

        # Define a constraint for node matching
        def constraint(pattern_node: ir.Node, graph_node: ir.Node) -> bool:
            return pattern_node.type == "any" or pattern_node.type == graph_node.type

        # Create and use the extractor
        extractor = SubgraphExtractor(pattern, node_constraint=constraint)
        new_impl, subgraph = extractor.extract(implementation)
        ```
        """
        self.pattern = pattern
        impl = cast(  # cast is ok, as graph will remain empty until extract is called
            ir.Implementation[GNode, ir.Edge],
            ir.Implementation(graph=pattern.graph.new()),
        )
        self._matcher = Matcher(
            pattern=pattern,
            graph=impl,
            node_constraint=node_constraint,
        )
        self.replacement = (
            pattern.graph.new()
            .add_edge("input", self.pattern.name)
            .add_edge(self.pattern.name, "output")
        )

    def extract(
        self, impl: ir.Implementation[GNode, ir.Edge]
    ) -> tuple[
        ir.Implementation[ir.Node, ir.Edge], ir.Implementation[ir.Node, ir.Edge]
    ]:
        """Extract a pattern from `impl` returning the new implementation and the extracted subgraph implementation.

        The subgraph will be replaced by a node with the same name/type as the pattern.
        Its `implementation` field will point to the newly extracted subgraph implementation.
        """
        interface = impl.graph.new().add_node("input").add_node("output")
        lhs = {"input": "input", "output": "output"}
        rhs = lhs
        self._matcher.graph = impl
        rewriter = g.GraphRewriter(
            replacement=self.replacement,
            match=self._matcher,
            pattern=self.pattern.graph,
            interface=interface,
            lhs=lhs,
            rhs=rhs,
        )
        result = rewriter.rewrite(impl.graph)
        subgraph = ir.Implementation(graph=impl.graph.new())
        subgraph_nodes = set(result.pattern_to_original.values())
        for node in subgraph_nodes:
            for successor in impl.successors(node).values():
                if successor.name in subgraph_nodes:
                    subgraph.add_edge(src=node, dst=successor.name)
            subgraph.add_node(impl.nodes[node])

        new_data = impl.data.copy()
        for node in subgraph_nodes - set(lhs.keys()):
            cast(dict[str, Any], new_data["nodes"]).pop(node)

        new_node_name = result.replacement_to_new[self.pattern.name]
        cast(dict[str, Any], new_data["nodes"])[new_node_name] = {
            "name": new_node_name,
            "type": self.pattern.data["type"],
            "implementation": self.pattern.name,
        }

        return ir.Implementation(data=new_data, graph=result.new_graph), subgraph
