from collections.abc import Callable

import elasticai.creator.graph as gr
from elasticai.creator import ir


class PatternNode(ir.Node):
    types: set[str]


class _MultiMatcher:
    def __init__(self, pattern: ir.Implementation[PatternNode, ir.Edge]) -> None:
        self._pattern = pattern
        self.impl: ir.Implementation[ir.Node, ir.Edge] = ir.Implementation(
            graph=gr.BaseGraph(),
        )
        self.matches: list[dict[str, str]] = []

    def node_constraint(self, pattern_node: str, graph_node: str) -> bool:
        pattern_types = self._pattern.nodes[pattern_node].types
        graph_type = self.impl.nodes[graph_node].type
        if "any" in pattern_types:
            return True
        fulfilled = graph_type in pattern_types
        return fulfilled

    def update_matches(self) -> None:
        self.matches = gr.find_all_subgraphs(
            pattern=self._pattern.graph,
            graph=self.impl.graph,
            node_constraint=self.node_constraint,
        )


def build_sequential_pattern(
    *pattern: tuple[str, set[str]],
) -> ir.Implementation[PatternNode, ir.Edge]:
    p = ir.Implementation(graph=gr.BaseGraph(), node_fn=PatternNode).add_node(
        name=pattern[0][0], data=dict(types=tuple(pattern[0][1]))
    )
    for (name1, _), (name2, types2) in zip(pattern, pattern[1:]):
        p.add_node(name=name2, data=dict(types=tuple(types2)))
        p.add_edge(src=name1, dst=name2, data={})
    return p


def _rewrite_sequential(
    original: ir.Implementation,
    pattern: ir.Implementation[PatternNode, ir.Edge],
    replacement: tuple[str, ...],
) -> tuple[gr.Graph[str], list[tuple[dict[str, str], dict[str, str]]]]:
    """exchange a sequential pattern for a sequential replacement.

    The nodes named "start" and "end" are considered interface nodes and will not be replaced.

    :param original: the original implementation
    :param pattern: a sequence of tuples with node names and their possible types. Use "any" to match any type.
    :param replacement: a sequence of node names to replace the pattern
    :return: the new graph and a mapping of the new node names to the original node names
    """
    lhs = dict(start="start", end="end")
    rhs = dict(start="start", end="end")
    mm = _MultiMatcher(pattern)

    repl = original.graph.new()
    for src, dst in zip(replacement, replacement[1:]):
        repl.add_edge(src, dst)
    mm.impl = original
    mm.update_matches()
    new_graph = original.graph
    matches: list[tuple[dict[str, str], dict[str, str]]] = []
    for match in gr.get_rewriteable_matches(original.graph, mm.matches, lhs.keys()):
        new_graph, _new_names = gr.rewrite(
            original=new_graph,
            match=match,
            replacement=repl,
            rhs=rhs,
            lhs=lhs,
        )
        matches.append((match, _new_names))
    return new_graph, matches


def _build_matched_impl(
    original: ir.Implementation[ir.Node, ir.Edge],
    pattern: ir.Implementation[PatternNode, ir.Edge],
    match: dict[str, str],
) -> ir.Implementation:
    matched_impl = ir.Implementation(graph=gr.BaseGraph())
    for pattern_node, original_node in match.items():
        matched_impl.add_node(
            name=pattern_node, data=original.nodes[original_node].data
        )
        for dst in pattern.successors(pattern_node):
            matched_impl.add_edge(src=pattern_node, dst=dst, data={})
        for src in pattern.predecessors(pattern_node):
            matched_impl.add_edge(src=src, dst=pattern_node, data={})
    return matched_impl


def move_pattern_to_subimpls(
    original: ir.Implementation,
    pattern: ir.Implementation[PatternNode, ir.Edge],
    basename: str,
    replacement_data_fn: Callable[[ir.Implementation], dict[str, ir.Attribute]],
    extracted_data_fn: Callable[
        [ir.Implementation], dict[str, dict[str, ir.Attribute]]
    ] = lambda _: {},
) -> list[ir.Implementation]:
    """Move all occurences of pattern into sub implementations.

    Each pattern match will be replaced by a single node with a name derived from the basename.
    For each match, we build a new implementation with the nodes and edges of the match.

    :param original: the original implementation
    :param pattern: Each node has a set of types that will be used as match constraints. Use "any" to match any type. The special node names `"start"` and `"end"` are considered interface nodes and will not be replaced.
    :param basename: the base name for the new nodes
    :param replacement_data_fn: a function that takes the matched implementation and returns a dictionary of attributes for the newly created node.
    :param extracted_data_fn: a function that takes the matched implementation and returns a dictionary of node names and their data for the extracted implementation. Use the names from the pattern to identify nodes.
    """
    new_graph, matches = _rewrite_sequential(
        original=original,
        pattern=pattern,
        replacement=("start", basename, "end"),
    )

    extracted_subgraphs = []
    new_impl = ir.Implementation(graph=new_graph)

    for match, repl_to_new_graph_names in matches:
        extracted_impl = ir.Implementation(graph=gr.BaseGraph())
        new_node_name = repl_to_new_graph_names[basename]
        implementation_name = f"{original.name}_{new_node_name}"
        matched_impl = _build_matched_impl(original, pattern, match)
        new_impl.add_node(
            name=new_node_name,
            data=replacement_data_fn(matched_impl)[new_node_name]  # type: ignore
            | {"implementation": implementation_name},  # type: ignore
        )
        extracted_impl.name = implementation_name
        extracted_impl.type = new_impl.nodes[new_node_name].type
        for pattern_node, original_node in match.items():
            if pattern_node == "start":
                extracted_impl.add_node(name="input", data={})
            elif pattern_node == "end":
                extracted_impl.add_node(name="output", data={})
            else:
                extracted_impl.add_node(
                    name=pattern_node, data=original.nodes[original_node].data
                )
            for dst in pattern.successors(pattern_node):
                extracted_impl.add_edge(src=pattern_node, dst=dst, data={})
            for src in pattern.predecessors(pattern_node):
                extracted_impl.add_edge(src=src, dst=pattern_node, data={})
        extracted_data = extracted_data_fn(matched_impl)
        for node, data in extracted_data.items():
            extracted_impl.nodes[node].data.update(data)
        extracted_subgraphs.append(extracted_impl)

    for node in new_impl.nodes:
        if node in original.nodes:
            new_impl.add_node(original.nodes[node])
    extracted_subgraphs.append(new_impl)
    return extracted_subgraphs
