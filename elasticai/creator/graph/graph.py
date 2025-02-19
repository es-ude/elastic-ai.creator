import warnings
from collections.abc import Iterable, Iterator, Set
from typing import Generic, TypeVar

T = TypeVar("T", int, str)


class Graph(Generic[T]):
    """All iterators in this class are guaranteed to have a fixed
    order. That means, the order of iteration is only allowed
    to change when the data structure is altered. If the
    content of two GraphDelegates is the same and they were built
    in the same way, then their iteration order is the same as well.

    NOTE: This class is not thread-safe.

    NOTE: We are not providing methods for removal of nodes or edges on purpose.
        If you need to remove nodes or edges, you should create a new GraphDelegate.
        Manipulation of the graph should usually be done in a dedicated build phase.
    """

    def __init__(self) -> None:
        """We keep successor and predecessor nodes just to allow for easier implementation.
        Currently, this implementation is not optimized for performance.
        """
        self.successors: dict[T, set[T]] = dict()
        self.predecessors: dict[T, set[T]] = dict()

    @staticmethod
    def from_dict(d: dict[T, Iterable[T]]):
        g: Graph[T] = Graph()
        for node, successors in d.items():
            g.add_node(node)
            for s in successors:
                g.add_edge(node, s)
        return g

    def as_dict(self) -> dict[T, set[T]]:
        return self.successors.copy()

    def add_edge(self, _from: T, _to: T):
        self.add_node(_from)
        self.add_node(_to)
        self.predecessors[_to].add(_from)
        self.successors[_from].add(_to)
        return self

    def add_node(self, node: T):
        if node not in self.predecessors:
            self.predecessors[node] = set()
        if node not in self.successors:
            self.successors[node] = set()
        return self

    @property
    def nodes(self) -> Set[T]:
        return self.successors.keys()

    def iter_nodes(self) -> Iterator[T]:
        """Iterator over nodes in a fixed but unspecified order."""
        yield from self.predecessors.keys()

    def iter_edges(self) -> Iterator[tuple[T, T]]:
        """Iterator over edges in a fixed but unspecified order."""
        for _from, _tos in self.successors.items():
            for _to in _tos:
                yield _from, _to

    def get_edges(self) -> Iterator[tuple[T, T]]:
        """Iterator over edges in a fixed but unspecified order."""
        warnings.warn(
            "get_edges() is deprecated, use iter_edges() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        yield from self.iter_edges()

    def get_successors(self, node: T) -> Iterator[T]:
        """Iterator over node successors in a fixed but unspecified order."""
        yield from sorted(self.successors[node])

    def get_predecessors(self, node: T) -> Iterator[T]:
        """Iterator over node predecessors in a fixed but unspecified order."""
        yield from sorted(self.predecessors[node])
