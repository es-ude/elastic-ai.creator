from collections.abc import Callable, Iterable, Iterator, Mapping
import copy
from functools import singledispatchmethod
from typing import Any, Generic, ParamSpec, Self, TypeVar, Union, overload

from .attribute import Attribute
from .core import Edge, Node
from .graph_delegate import GraphDelegate
from .graph_iterators import (
    bfs_iter_down,
    bfs_iter_up,
    dfs_pre_order,
)
from .ir_data import IrData
from .required_field import read_only_field

N = TypeVar("N", bound=Node)
E = TypeVar("E", bound=Edge)

StoredT = TypeVar("StoredT", bound=Attribute)
VisibleT = TypeVar("VisibleT")
P = ParamSpec("P")


class Graph(IrData, Generic[N, E], create_init=False):
    __slots__ = ("data", "_g", "_node_data", "_edge_data", "_node_fn", "_edge_fn")

    @overload
    def __init__(
        self: "Graph[N, E]",
        *,
        node_fn: Callable[[dict], N],
        edge_fn: Callable[[dict], E],
        nodes: Iterable[N] = tuple(),
        edges: Iterable[E] = tuple(),
        data: dict | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Graph[Node, Edge]",
        *,
        nodes: Iterable[Node] = tuple(),
        edges: Iterable[Edge] = tuple(),
        data: dict | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Graph[N, Edge]",
        *,
        node_fn: Callable[[dict], N],
        nodes: Iterable[N] = tuple(),
        edges: Iterable[Edge] = tuple(),
        data: dict | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Graph[Node, E]",
        *,
        edge_fn: Callable[[dict], E],
        nodes: Iterable[Node] = tuple(),
        edges: Iterable[E] = tuple(),
        data: dict | None = None,
    ): ...

    def __init__(
        self,
        *,
        node_fn=Node,
        edge_fn=Edge,
        nodes=tuple(),
        edges=tuple(),
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
        self._node_data: dict[str, Attribute] = data["nodes"]
        self._edge_data: dict[tuple[str, str], Attribute] = data["edges"]
        for n in self._node_data:
            self._g.add_node(n)
        for src, sink in self._edge_data:
            self._g.add_edge(src, sink)
        self.add_nodes(nodes)
        self.add_edges(edges)
        self._node_fn = node_fn
        self._edge_fn = edge_fn

    @overload
    def add_node(self, n: N) -> Self: ...

    @overload
    def add_node(self, n: Node) -> Self: ...

    @overload
    def add_node(self, n: dict[str, Attribute]) -> Self: ...

    @overload
    def add_node(self, *, name: str, type: str, **attributes: Attribute) -> Self: ...

    @singledispatchmethod  # type: ignore[operator, misc]
    def add_node(self, *args, **kwargs) -> Self:
        raise NotImplementedError()

    @add_node.register  # type: ignore[attr-defined]
    def _(self, n: Node) -> Self:
        self.add_node(n.data)
        return self

    @add_node.register  # type: ignore[attr-defined]
    def _(self, n: dict) -> Self:
        self._g.add_node(n["name"])
        self._node_data[n["name"]] = n
        return self

    @add_node.register  # type: ignore[attr-defined]
    def _(self, *, name: str, type: str, **attributes: Attribute) -> Self:
        n = {"name": name, "type": type, **attributes}
        self.add_node(n)
        return self

    @overload
    def add_edge(self, e: E) -> Self: ...

    @overload
    def add_edge(self, e: Edge) -> Self: ...

    @overload
    def add_edge(self, e: dict) -> Self: ...

    @overload
    def add_edge(self, *, src: str, sink: str, **attributes: Attribute) -> Self: ...

    @singledispatchmethod  # type: ignore[operator, misc]
    def add_edge(self, *args, **kwargs) -> Self:
        raise NotImplementedError()

    @add_edge.register  # type: ignore[attr-defined]
    def _(self, e: Edge) -> Self:
        self.add_edge(e.data)
        return self

    @add_edge.register  # type: ignore[attr-defined]
    def _(self, e: dict) -> Self:
        self._g.add_edge(e["src"], e["sink"])
        self._edge_data[(e["src"], e["sink"])] = e
        return self

    @add_edge.register  # type: ignore[attr-defined]
    def _(self, *, src: str, sink: str, **attributes: Attribute) -> Self:
        e = {"src": src, "sink": sink, **attributes}
        self.add_edge(e)
        return self

    def add_nodes(self, ns: Iterable[N | Node | dict[str, Attribute]]) -> Self:
        for n in ns:
            self.add_node(n)
        return self

    def add_edges(self, es: Iterable[E | Edge | dict[str, Attribute]]) -> Self:
        for e in es:
            self.add_edge(e)
        return self

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

    @read_only_field
    def nodes(self, _: dict[str, Attribute]) -> Mapping[str, N]:
        return self._get_node_mapping(self._g.iter_nodes)

    @read_only_field
    def edges(self, _: dict[str, Attribute]) -> Mapping[tuple[str, str], E]:
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

    def as_dict(self) -> dict[str, Attribute]:
        data = self.data.copy()
        data["nodes"] = list(n.data for n in self.nodes.values())
        data["edges"] = list(e.data for e in self.edges.values())
        return data

    @classmethod
    @overload
    def from_dict(
        cls,
        d: dict[str, Any],
    ) -> "Graph[Node, Edge]": ...

    @classmethod
    @overload
    def from_dict(
        cls,
        d: dict[str, Any],
        node_fn: Callable[[dict], N],
    ) -> "Graph[N, Edge]": ...

    @classmethod
    @overload
    def from_dict(
        cls,
        d: dict[str, Any],
        node_fn: Callable[[dict], N],
        edge_fn: Callable[[dict], E],
    ) -> "Graph[N, E]": ...

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        node_fn: Callable[[dict], N] | Callable[[dict], Node] = Node,
        edge_fn: Callable[[dict], E] | Callable[[dict], Edge] = Edge,
    ) -> "Union[Graph[N, Edge], Graph[N, E], Graph[Node, Edge]]":
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

        :::{note}
        `data` is not copied, so it is shared between the original and the new graph.
        :::
        """
        self.data = data

    def get_empty_copy(self) -> "Graph[N, E]":
        """Get an empty version of the graph, with the same node_fn and edge_fn.

        This is the complement of `load_dict`.

        :::{note}
        `node_fn` and `edge_fn` are deep copied, so they are not shared between the
        original and the new graph.
        :::
        """
        return self.__class__(
            node_fn=copy.deepcopy(self._node_fn), edge_fn=copy.deepcopy(self._edge_fn)
        )


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
