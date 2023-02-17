import unittest
from typing import Callable, Iterable, Iterator, Reversible

from elasticai.creator.tests.unit.vhdl.components.test_signal_connector import first
from elasticai.creator.vhdl.dataflow.node_iteration import ancestors_breadth_first
from elasticai.creator.vhdl.graph import Graph, Node


class AncestorsInBreadthFirstOrderTest(unittest.TestCase):
    """
    Test:
      - do not visit grandparent twice
    """

    def check_graph_traversal(
        self,
        node: Node,
        expected_ids: list[int],
        traverser: Callable[[Node], Iterator[Node]],
    ):
        ids: list[str] = []
        for n in traverser(node):
            ids.append(n.id())
        self.assertEqual(list(map(str, expected_ids)), ids)

    def test_yield_210(self) -> None:
        edges = ((0, 1), (1, 2), (2, 3))
        graph = BasicGraph(edges)
        node = first(reversed(graph.nodes))
        self.check_graph_traversal(node, [2, 1, 0], ancestors_breadth_first)

    def test_yield_420(self) -> None:
        edges = ((0, 1), (0, 2), (1, 3), (2, 4), (4, 5))
        graph = BasicGraph(edges)
        node = next(reversed(graph.nodes))
        self.check_graph_traversal(node, [4, 2, 0], ancestors_breadth_first)

    def test_yield_420_for_multiparent_node_2(self) -> None:
        edges = ((0, 2), (4, 5), (2, 5))
        graph = BasicGraph(edges)
        node = next(reversed(graph.nodes))
        self.check_graph_traversal(node, [4, 2, 0], ancestors_breadth_first)

    def test_visit_grandparent_only_once(self) -> None:
        edges = ((0, 1), (0, 2), (1, 3), (2, 3))
        graph = BasicGraph(edges)
        node = next(reversed(graph.nodes))
        expected_sequence_in_correct_order = [1, 2, 0]
        self.check_graph_traversal(
            node, expected_sequence_in_correct_order, ancestors_breadth_first
        )

    def test_visit_two_levels_breadth_first(self) -> None:
        edges = ((0, 3), (1, 3), (0, 4), (2, 4), (3, 5), (4, 5))
        graph = BasicGraph(edges)
        node = next(reversed(graph.nodes))
        expected_sequence_in_correct_order = [3, 4, 0, 1, 2]
        self.check_graph_traversal(
            node, expected_sequence_in_correct_order, ancestors_breadth_first
        )


class BasicNode(Node):
    def __init__(self, id: int):
        self._parents: list["BasicNode"] = []
        self._children: list["BasicNode"] = []
        self._id = id

    def id(self) -> str:
        return f"{self._id}"

    @property
    def parents(self) -> Iterable["BasicNode"]:
        return self._parents

    @property
    def children(self) -> Iterable["BasicNode"]:
        return self._children

    def append(self, child: "BasicNode") -> None:
        self._children.append(child)
        child._parents.append(self)

    def __repr__(self):
        return f"Node({self.id()})"


class BasicGraph(Graph):
    def __init__(self, edges: Iterable[tuple[int, int]]):
        number_of_nodes = max((node for edge in edges for node in edge)) + 1
        self._nodes = [BasicNode(n) for n in range(number_of_nodes)]
        node_edges = ((self._nodes[e[0]], self._nodes[e[1]]) for e in edges)
        for _from, _to in node_edges:
            _from.append(_to)

    @property
    def nodes(self) -> Reversible["BasicNode"]:
        return self._nodes
