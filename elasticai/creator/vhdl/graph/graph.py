from typing import Iterable, Iterator, Reversible, TypeVar

from elasticai.creator.vhdl.graph.typing import Graph, Node, T_Node


class NodeGraph(Graph[T_Node]):
    def __init__(self: Graph[T_Node], root: T_Node):
        self._root = root

    @property
    def nodes(self) -> Reversible[T_Node]:
        return NodeTraversal(self._root)


class NodeTraversal(Reversible[T_Node]):
    @staticmethod
    def _visit_nodes_forwards(node: "T_Node") -> Iterator["T_Node"]:
        yield node
        for child in node.children:
            yield from NodeTraversal._visit_nodes_forwards(child)

    def __reversed__(self) -> Iterator[T_Node]:
        nodes = list(self)
        yield from reversed(nodes)

    def __iter__(self: "NodeTraversal[T_Node]") -> Iterator["T_Node"]:
        yield from self._visit_nodes_forwards(self._root)

    def __init__(self: "NodeTraversal[T_Node]", root: T_Node):
        self._root = root


def ancestors_breadth_first(node: T_Node) -> Iterator[T_Node]:
    visited: set[T_Node] = set()

    def _iterator(node: T_Node) -> Iterator[T_Node]:
        for p in node.parents:
            if p not in visited:
                visited.add(p)
                yield p
        for p in node.parents:
            yield from _iterator(p)

    yield from _iterator(node)


T_BasicNode_co = TypeVar("T_BasicNode_co", bound="BasicNode", covariant=True)


class BasicNode(Node):
    def __init__(self: T_BasicNode_co, id: int):
        self._parents: list[Node] = []
        self._children: list[Node] = []
        self._id = id

    def id(self) -> str:
        return f"{self._id}"

    @property
    def parents(self) -> Iterable[Node]:
        return self._parents

    @property
    def children(self) -> Iterable[Node]:
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
