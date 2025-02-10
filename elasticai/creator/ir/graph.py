from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Generic, TypeVar

from .core import Edge, Node
from .graph_delegate import GraphDelegate
from .graph_iterators import (
    bfs_iter_down,
    bfs_iter_up,
    dfs_pre_order,
)
from .ir_data import IrData

N = TypeVar("N", bound=Node)
E = TypeVar("E", bound=Edge)


class Graph(IrData, Generic[N, E], create_init=False):
    __slots__ = ("data", "_g", "_node_data", "_edge_data", "_node_fn", "_edge_fn")

    def __init__(
        self,
        *,
        node_fn: Callable[[dict], N] = Node,
        edge_fn: Callable[[dict], E] = Edge,
        nodes: Iterable[N] = tuple(),
        edges: Iterable[E] = tuple(),
        data=None,
    ) -> None:
        if data is None:
            data = {}
        if "nodes" not in data:
            data["nodes"] = {}
        if "edges" not in data:
            data["edges"] = {}
        super().__init__(data)
        self._g: GraphDelegate[str] = GraphDelegate()
        self._node_data = data["nodes"]
        self._edge_data = data["edges"]
        for n in self._node_data:
            self._g.add_node(n)
        for src, sink in self._edge_data:
            self._g.add_edge(src, sink)
        for n in nodes:
            self.add_node(n)
        for e in edges:
            self.add_edge(e)
        self._node_fn = node_fn
        self._edge_fn = edge_fn

    def add_node(self, n: N) -> None:
        self._g.add_node(n.name)
        self._node_data[n.name] = n.data

    def add_nodes(self, ns: Iterable[N]) -> None:
        for n in ns:
            self.add_node(n)

    def add_edges(self, es: Iterable[E]) -> None:
        for e in es:
            self.add_edge(e)

    def add_edge(self, e: E) -> None:
        self._g.add_edge(e.src, e.sink)
        self._edge_data[(e.src, e.sink)] = e.data

    def successors(self, node: str | N) -> Mapping[str, N]:
        if not isinstance(node, str):
            node = node.name

        def _successors():
            return self._g.get_successors(node)

        return self._get_node_mapping(_successors)

    def predecessors(self, node: str | N) -> Mapping[str, N]:
        if not isinstance(node, str):
            node = node.name

        def _predecessors():
            return self._g.get_predecessors(node)

        return self._get_node_mapping(_predecessors)

    def _get_node_mapping(self, keys: Callable[[], Iterable[str]]) -> Mapping[str, N]:
        return _ReadOnlyMappingInOrderAsIterable(keys, self._node_data, self._node_fn)

    def _get_edge_mapping(
        self, keys: Callable[[], Iterable[tuple[str, str]]]
    ) -> Mapping[tuple[str, str], E]:
        return _ReadOnlyMappingInOrderAsIterable(keys, self._edge_data, self._edge_fn)

    @property
    def nodes(self) -> Mapping[str, N]:
        return self._get_node_mapping(self._g.iter_nodes)

    @property
    def edges(self) -> Mapping[tuple[str, str], E]:
        return self._get_edge_mapping(self._g.iter_edges)

    def iter_bfs_down_from(self, node: str) -> Mapping[str, N]:
        def iter_keys():
            return bfs_iter_down(self._g.get_successors, node)

        return self._get_node_mapping(iter_keys)

    def iter_bfs_up_from(self, node: str) -> Mapping[str, N]:
        def iter_keys():
            return bfs_iter_up(self._g.get_predecessors, self._g.get_successors, node)

        return self._get_node_mapping(iter_keys)

    def iter_dfs_preorder_down_from(self, node: str) -> Mapping[str, N]:
        def iter_keys():
            return dfs_pre_order(self._g.get_successors, node)

        return self._get_node_mapping(iter_keys)

    def as_dict(self) -> dict:
        data = self.data.copy()
        data["nodes"] = list(data["nodes"].values())
        data["edges"] = list(data["edges"].values())
        return data

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        node_fn: Callable[[dict], N] = Node,
        edge_fn: Callable[[dict], E] = Edge,
    ) -> "Graph[N, E]":
        data = d.copy()
        data["nodes"] = {node["name"]: node for node in data["nodes"]}
        data["edges"] = {(edge["src"], edge["sink"]): edge for edge in data["edges"]}
        g = cls(node_fn=node_fn, edge_fn=edge_fn, data=data)
        return g

    def load_dict(self, data: dict[str, Any]) -> None:
        """override the current state with `data`.

        Use this if you want to change the underlying data structure
        for an already existing graph, e.g., because you want to reuse
        the set `node_fn`, `edge_fn` functions for constructing
        nodes and edges.
        """
        self.data = data


_K = TypeVar("_K")
_V = TypeVar("_V")


class _ReadOnlyMappingInOrderAsIterable(Mapping[_K, _V]):
    def __init__(
        self,
        iterable: Callable[[], Iterable[_K]],
        d: dict[_K, Any],
        value_constructor: Callable[[Any], _V],
    ):
        self._iterable = iterable
        self._d = d
        self._value_constructor = value_constructor

    def __iter__(self) -> Iterator[_K]:
        yield from self._iterable()

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, k: object) -> bool:
        return k in self._d

    def __getitem__(self, k: _K) -> _V:
        return self._value_constructor(self._d[k])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self) == dict(other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self)})"
