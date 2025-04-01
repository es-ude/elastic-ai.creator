from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Mapping, Set
from typing import Self

from .graph import Graph


# we have to keep this bound to str, until we move to python 3.13 with default type arguments
class BaseGraph(Graph[str]):
    """All iterators in this class are guaranteed to have a fixed
    order. strhat means, the order of iteration is only allowed
    to change when the data structure is altered. If the
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
        self._successors: dict[str, set[str]] = dict()
        self._predecessors: dict[str, set[str]] = dict()

    @property
    def successors(self) -> Mapping[str, set[str]]:
        return self._successors.keys().mapping

    @property
    def predecessors(self) -> Mapping[str, set[str]]:
        return self._predecessors.keys().mapping

    @staticmethod
    def from_dict(d: dict[str, Iterable[str]]):
        g: Graph[str] = BaseGraph()
        for node, successors in d.items():
            g.add_node(node)
            for s in successors:
                g.add_edge(node, s)
        return g

    def as_dict(self) -> dict[str, set[str]]:
        return self._successors.copy()

    def add_edge(self, _from: str, _to: str) -> Self:
        self.add_node(_from)
        self.add_node(_to)
        self.predecessors[_to].add(_from)
        self.successors[_from].add(_to)
        return self

    def add_node(self, node: str) -> Self:
        if node not in self.predecessors:
            self._predecessors[node] = set()
        if node not in self.successors:
            self._successors[node] = set()
        return self

    @property
    def nodes(self) -> Set[str]:
        return self._successors.keys()

    def iter_nodes(self) -> Iterator[str]:
        """Iterator over nodes in a fixed but unspecified order."""
        yield from self.predecessors.keys()

    def iter_edges(self) -> Iterator[tuple[str, str]]:
        """Iterator over edges in a fixed but unspecified order."""
        for _from, _tos in self.successors.items():
            for _to in _tos:
                yield _from, _to

    def get_edges(self) -> Iterator[tuple[str, str]]:
        """Iterator over edges in a fixed but unspecified order."""
        warnings.warn(
            "get_edges() is deprecated, use iter_edges() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        yield from self.iter_edges()

    def get_successors(self, node: str) -> Iterator[str]:
        """Iterator over node successors in a fixed but unspecified order."""
        yield from sorted(self.successors[node])

    def get_predecessors(self, node: str) -> Iterator[str]:
        """Iterator over node predecessors in a fixed but unspecified order."""
        yield from sorted(self.predecessors[node])

    def new(self) -> Graph[str]:
        return BaseGraph()
