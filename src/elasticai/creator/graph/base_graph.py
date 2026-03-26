import copy
import warnings
from collections.abc import Collection, Iterable, Iterator, Mapping, Set
from typing import Any, Hashable, Self, TypeVar, cast

from .graph import Graph

T = TypeVar("T", bound=Hashable)  # noqa: F821


# we have to keep this bound to T, until we move to python 3.13 with default type arguments
class BaseGraph(Graph[T]):
    """All iterators in this class are guaranteed to have a fixed
    order. That means, the order of iteration is only allowed
    to change when the data Tucture is altered. If the
    content of two GraphDelegates is the same and they were built
    in the same way, then their iteration order is the same as well.

    :::{caution}
    This class is not thread-safe.
    :::

    :::{note}
    We are not providing methods for removal of nodes or edges on purpose.
        If you need to remove nodes or edges, you should create a new GraphDelegate.
        Manipulation of the graph should usually be done in a dedicated build phase.
    :::
    """

    def __init__(self) -> None:
        """We keep successor and predecessor nodes just to allow for easier implementation.
        Currently, this implementation is not optimized for performance.
        """
        self._successors: dict[T, dict[T, Any]] = dict()
        self._predecessors: dict[T, dict[T, Any]] = dict()

    @property
    def successors(self) -> Mapping[T, Collection[T]]:
        return self._successors.keys().mapping

    @property
    def predecessors(self) -> Mapping[T, Collection[T]]:
        return self._predecessors.keys().mapping

    @staticmethod
    def from_dict(d: dict[T, Iterable[T]]):
        g: Graph[T] = BaseGraph()
        for node, successors in d.items():
            g.add_node(node)
            for s in successors:
                g.add_edge(node, s)
        return g

    def as_dict(self) -> dict[T, dict[T, Any]]:
        return self._successors.copy()

    def add_edge(self, src: T, dst: T, /) -> Self:
        def add_to_adj_map(adj, a, b):
            adj[a][b] = None

        self.add_node(src)
        self.add_node(dst)
        add_to_adj_map(self._predecessors, dst, src)
        add_to_adj_map(self._successors, src, dst)
        return self

    def add_node(self, node: T) -> Self:
        def add_to_adj(adj, n):
            if n not in adj:
                adj[n] = {}

        add_to_adj(self._predecessors, node)
        add_to_adj(self._successors, node)
        return self

    @property
    def nodes(self) -> Set[T]:
        return self._successors.keys()

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
        yield from self.successors[node]

    def get_predecessors(self, node: T) -> Iterator[T]:
        """Iterator over node predecessors in a fixed but unspecified order."""
        yield from self.predecessors[node]

    def new(self: Self) -> Self:
        return cast(Self, BaseGraph())

    def copy(self: Self) -> Self:
        g = self.new()
        g._predecessors = copy.deepcopy(self._predecessors)
        g._successors = copy.deepcopy(self._successors)
        return g
