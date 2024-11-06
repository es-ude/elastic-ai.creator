from collections.abc import Hashable, Iterable, Iterator
from typing import Generic, TypeVar

HashableT = TypeVar("HashableT", bound=Hashable)


class GraphDelegate(Generic[HashableT]):
    def __init__(self) -> None:
        """We keep successor and predecessor nodes just to allow for easier implementation.
        Currently, this implementation is not optimized for performance.
        """
        self.successors: dict[HashableT, dict[HashableT, None]] = dict()
        self.predecessors: dict[HashableT, dict[HashableT, None]] = dict()

    @staticmethod
    def from_dict(d: dict[HashableT, Iterable[HashableT]]):
        g = GraphDelegate()
        for node, successors in d.items():
            for s in successors:
                g.add_edge(node, s)
        return g

    def as_dict(self) -> dict[HashableT, set[HashableT]]:
        return self.successors.copy()

    def add_edge(self, _from: HashableT, _to: HashableT):
        self.add_node(_from)
        self.add_node(_to)
        self.predecessors[_to][_from] = None
        self.successors[_from][_to] = None
        return self

    def add_node(self, node: HashableT):
        if node not in self.predecessors:
            self.predecessors[node] = dict()
        if node not in self.successors:
            self.successors[node] = dict()
        return self

    def iter_nodes(self) -> Iterator[HashableT]:
        yield from self.predecessors.keys()

    def get_edges(self) -> Iterator[tuple[HashableT, HashableT]]:
        for _from, _tos in self.successors.items():
            for _to in _tos:
                yield _from, _to

    def get_successors(self, node: HashableT) -> Iterator[HashableT]:
        yield from self.successors[node]

    def get_predecessors(self, node: HashableT) -> Iterator[HashableT]:
        yield from self.predecessors[node]
