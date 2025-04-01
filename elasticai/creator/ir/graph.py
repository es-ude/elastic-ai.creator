from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Generic, TypeVar

from .core import Edge, Node
from .graph_delegate import GraphDelegate

N = TypeVar("N", bound=Node)
E = TypeVar("E", bound=Edge)
Self = TypeVar("Self", bound="Graph")


class Graph(Generic[N, E]):
    def __init__(
        self: Self, nodes: Iterable[N] = tuple(), edges: Iterable[E] = tuple()
    ) -> None:
        self._g: GraphDelegate[str] = GraphDelegate()
        self._edge_data: dict[tuple[str, str], E] = dict()
        self._node_data: dict[str, N] = dict()
        self.add_edges(edges)
        self.add_nodes(nodes)

    def add_node(self: Self, n: N) -> None:
        self._g.add_node(n.name)
        self._node_data[n.name] = n

    def add_nodes(self: Self, ns: Iterable[N]) -> None:
        for n in ns:
            self.add_node(n)

    def add_edges(self: Self, es: Iterable[E]) -> None:
        for e in es:
            self.add_edge(e)

    def add_edge(self: Self, e: Edge) -> None:
        self._g.add_edge(e.src, e.sink)
        self._edge_data[(e.src, e.sink)] = e

    def successors(self: Self, node: str | N) -> Mapping[str, N]:
        if not isinstance(node, str):
            node = node.name

        def _successors():
            return self._g.get_successors(node)

        return _ReadOnlyMappingInOrderAsIterable(_successors, self._node_data)

    def predecessors(self: Self, node: str | N) -> Mapping[str, N]:
        if not isinstance(node, str):
            node = node.name

        def _predecessors():
            return self._g.get_predecessors(node)

        return _ReadOnlyMappingInOrderAsIterable(_predecessors, self._node_data)

    @property
    def nodes(self: Self) -> Mapping[str, N]:
        return _ReadOnlyMappingInOrderAsIterable(self._g.iter_nodes, self._node_data)

    @property
    def edges(self: Self) -> Mapping[tuple[str, str], E]:
        return _ReadOnlyMappingInOrderAsIterable(self._g.get_edges, self._edge_data)


_K = TypeVar("_K")
_V = TypeVar("_V")


class _ReadOnlyMappingInOrderAsIterable(Mapping[_K, _V]):
    def __init__(self, iterable: Callable[[], Iterator[_K]], d: dict[_K, _V]):
        self._iterable = iterable
        self._d = d

    def __iter__(self) -> Iterator[_K]:
        yield from self._iterable()

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, k: object) -> bool:
        return k in self._d

    def __getitem__(self, k: _K) -> _V:
        return self._d[k]
