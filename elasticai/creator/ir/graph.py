from collections.abc import Iterator

from .attribute import AttributeT
from .edge import Edge
from .humble_base_graph import HumbleBaseGraph
from .node import Node


class Graph:
    def __init__(self):
        self._hbg: HumbleBaseGraph[str] = HumbleBaseGraph()
        self._nodes: dict[str, Node] = {}
        self._edges: dict[(str, str), Edge] = dict()
        self.add_node(Node("input", "input"))
        self.add_node(Node("output", "output"))

    def get_node(self, name: str) -> Node:
        return self._nodes[name]

    def has_node(self, name: str) -> bool:
        return name in self._nodes

    def add_node(self, n: Node) -> "Graph":
        self._nodes[n.name] = n
        self._hbg.add_node(n.name)
        return self

    def add_edge(self, e: Edge) -> "Graph":
        self._hbg.add_edge(e.src, e.sink)
        self._edges[(e.src, e.sink)] = e
        return self

    def iter_edges(self) -> Iterator[Edge]:
        yield from self._edges.values()

    def iter_src_sink_pairs(self) -> Iterator[tuple[Node, Node]]:
        for src, sink in self._edges:
            yield self.get_node(src), self.get_node(sink)

    def iter_src_sink_name_pairs(self) -> Iterator[tuple[str, str]]:
        yield from self._edges

    def iter_nodes(self) -> Iterator[Node]:
        for n in self._hbg.iter_nodes():
            yield self._nodes[n]

    def iter_node_names(self) -> Iterator[str]:
        yield from self._hbg.iter_nodes()

    def get_nodes_by_type(self, type: str) -> Iterator[Node]:
        for n in self.iter_nodes():
            if n.type == type:
                yield n

    def get_successors(self, name: str) -> Iterator[Node]:
        for name in self._hbg.get_successors(name):
            yield self._nodes[name]

    def get_successor_names(self, name: str) -> Iterator[str]:
        yield from self._hbg.get_successors(name)

    def get_predecessors(self, name: str) -> Iterator[Node]:
        for name in self._hbg.get_predecessors(name):
            yield self._nodes[name]

    def as_dict(self) -> dict[str, AttributeT]:
        def make_dict(x):
            return x.as_dict()

        return dict(
            nodes=list(map(make_dict, self.iter_nodes())),
            edges=list(map(make_dict, self.iter_edges())),
        )

    @classmethod
    def from_dict(cls, data: dict[str, AttributeT]) -> "Graph":
        g = cls()
        for n in data["nodes"]:
            g.add_node(Node.from_dict(n))
        for e in data["edges"]:
            g.add_edge(Edge.from_dict(e))
