from typing import Iterator, Reversible

from elasticai.creator.vhdl.data_path_connection.typing import T_Node


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
