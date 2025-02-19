from collections.abc import Callable
from typing import Iterable

from elasticai.creator import graph as g


class Graph:
    def __init__(self, wrapped: g.Graph[str], data: dict[str, str]):
        self.wrapped = wrapped
        self.data = data

    @property
    def nodes(self):
        return self.wrapped.nodes

    @property
    def edges(self):
        return list(self.wrapped.iter_edges())

    def as_dict(self):
        return {"nodes": set(self.nodes), "edges": set(self.edges)}


def build_graph_from_dict(
    d: dict[tuple[str, str], list[str]],
) -> Graph:
    graph_dict: dict[str, Iterable[str]] = {k[0]: v for k, v in d.items()}
    data_dict = {k[0]: k[1] for k in d}

    return Graph(g.Graph.from_dict(graph_dict), data=data_dict)


def find_matches(graph, pattern) -> list[dict[str, str]]:
    def node_constraint(graph_node, pattern_node):
        return graph.data[graph_node] == pattern.data[pattern_node]

    return g.find_subgraphs(graph.wrapped, pattern.wrapped, node_constraint)


class Matcher:
    def __init__(self, pattern: Graph):
        self.pattern = pattern
        self.graph: None | Graph = None

    def set_graph(self, graph: Graph):
        self.graph = graph

    def _node_constraint(self, graph_node: str, pattern_node: str) -> bool:
        assert self.graph is not None
        return self.graph.data[graph_node] == self.pattern.data[pattern_node]

    def __call__(
        self, graph: g.Graph[str], pattern: g.Graph[str]
    ) -> list[dict[str, str]]:
        return g.find_subgraphs(graph, pattern, self._node_constraint)


def get_rewriter(
    pattern: Graph,
    interface: Graph,
    replacement: Graph,
    lhs: dict[str, str],
    rhs: dict[str, str],
) -> Callable[[Graph], Graph]:
    match = Matcher(pattern)

    rewriter = g.GraphRewriter(
        pattern=pattern.wrapped,
        interface=interface.wrapped,
        replacement=replacement.wrapped,
        match=match,
        lhs=lhs,
        rhs=rhs,
    )

    def rewrite(graph: Graph):
        match.graph = graph
        return Graph(rewriter.rewrite(graph.wrapped), {})

    return rewrite
