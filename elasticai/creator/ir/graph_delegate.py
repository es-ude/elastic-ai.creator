from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

_T = TypeVar("_T", int, str)


class GraphDelegate(Generic[_T]):
    def __init__(self) -> None:
        """We keep successor and predecessor nodes just to allow for easier implementation.
        Currently, this implementation is not optimized for performance.
        """
        self.successors: dict[_T, set[_T]] = dict()
        self.predecessors: dict[_T, set[_T]] = dict()

    @staticmethod
    def from_dict(d: dict[_T, Iterable[_T]]):
        g: GraphDelegate[_T] = GraphDelegate()
        for node, successors in d.items():
            for s in successors:
                g.add_edge(node, s)
        return g

    def as_dict(self) -> dict[_T, set[_T]]:
        return self.successors.copy()

    def add_edge(self, _from: _T, _to: _T):
        self.add_node(_from)
        self.add_node(_to)
        self.predecessors[_to].add(_from)
        self.successors[_from].add(_to)
        return self

    def add_node(self, node: _T):
        if node not in self.predecessors:
            self.predecessors[node] = set()
        if node not in self.successors:
            self.successors[node] = set()
        return self

    def iter_nodes(self) -> Iterator[_T]:
        yield from self.predecessors.keys()

    def get_edges(self) -> Iterator[tuple[_T, _T]]:
        for _from, _tos in self.successors.items():
            for _to in _tos:
                yield _from, _to

    def get_successors(self, node: _T) -> Iterator[_T]:
        yield from sorted(self.successors[node])

    def get_predecessors(self, node: _T) -> Iterator[_T]:
        yield from sorted(self.predecessors[node])
