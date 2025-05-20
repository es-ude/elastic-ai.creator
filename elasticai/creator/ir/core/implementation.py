from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import (
    Any,
    Generic,
    MutableMapping,
    ParamSpec,
    Protocol,
    Self,
    TypeVar,
    cast,
    overload,
)

from elasticai.creator.graph import Graph
from elasticai.creator.ir.base import Attribute, IrData, read_only_field

from .core import Edge, Node

N = TypeVar("N", bound=Node, covariant=True)
E = TypeVar("E", bound=Edge, covariant=True)

StoredT = TypeVar("StoredT", bound=Attribute)
VisibleT = TypeVar("VisibleT")
P = ParamSpec("P")


class NodeFn(Protocol[N]):
    def __call__(self, name: str, data: dict[str, Attribute]) -> N: ...


class EdgeFn(Protocol[E]):
    def __call__(self, src: str, dst: str, data: dict[str, Attribute]) -> E: ...


class Implementation(IrData, Generic[N, E], create_init=False):
    __slots__ = (
        "data",
        "graph",
        "_node_fn",
        "_edge_fn",
        "_nodes",
        "_edges",
    )

    name: str
    type: str

    @overload
    def __init__(
        self: "Implementation[N, E]",
        *,
        graph: Graph[str],
        edge_fn: EdgeFn[E],
        node_fn: NodeFn[N],
        data: dict[str, Attribute] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Implementation[N, Edge]",
        *,
        graph: Graph,
        node_fn: NodeFn[N],
        data: dict[str, Attribute] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Implementation[Node, E]",
        *,
        graph: Graph[str],
        edge_fn: EdgeFn[E],
        data: dict[str, Attribute] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Implementation[Node, Edge]",
        *,
        graph: Graph[str],
        data: dict[str, Attribute] | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        graph: Graph[str],
        node_fn: NodeFn = Node,
        edge_fn: EdgeFn = Edge,
        data: dict[str, Attribute] | None = None,
    ) -> None:
        """Create a new Implementation. Nodes and edges from `data` will not be automatically added to the graph.

        The constructor will combine `graph` and `data` so we can access nodes and edges of type `N` and `E` respectively.
        And use these to manipulate the underlying graph structure.
        It is the callers responsibility to keep `graph` and `data` in sync.
        This might seem like a limitation at first, but in fact it is a feature.
        It allows us to combine existing graph structures with new underlying data.
        A common use case for this is to handle subgraphs as in the following example:

        ```python
        subgraph = Graph()

        for node in original_impl.nodes.values():
            if fulfills_constraint(node):
                subgraph.add_node(node.name)

        for src, dst in original_impl.edges:
            if fulfills_constraint(original_impl.nodes[src]) and fulfills_constraint(original_impl.nodes[dst]):
                subgraph.add_edge(src, dst)

        new_impl = Implementation(graph=subgraph, data=original_impl.data)

        for node in new_impl.nodes.values():  # now we can iterate over the nodes of the subgraph
            do_something(node)
        ```
        """
        if data is None:
            data = {"nodes": {}, "edges": {}}
        if "nodes" not in data:
            data["nodes"] = {}
        if "edges" not in data:
            data["edges"] = {}
        super().__init__(data)

        self.graph = graph
        self._nodes: dict[str, Attribute] = cast(
            dict[str, Attribute], self.data["nodes"]
        )
        self._edges: dict[str, dict[str, Attribute]] = cast(
            dict[str, dict[str, Attribute]], self.data["edges"]
        )

        self._node_fn = node_fn
        self._edge_fn = edge_fn

    @property
    def _node_data(self) -> dict[str, Attribute]:
        return cast(dict[str, Attribute], self.data["nodes"])

    @property
    def _edge_data(self) -> MutableMapping[tuple[str, str], Attribute]:
        return _NestedDictToTupleKeyAdapter(
            cast(dict[str, dict[str, "Attribute"]], self.data["edges"])
        )

    @overload
    def add_node(self, n: Node) -> Self: ...

    @overload
    def add_node(self, *, name: str, data: dict[str, Attribute]) -> Self: ...

    def add_node(self, *args, **kwargs) -> Self:
        if len(args) + len(kwargs) == 1:
            bound = _bind_args(["node"], {}, *args, **kwargs)
            node = bound["node"]
            return self._add_node(node.name, node.data)
        else:
            bound = _bind_args(["name", "data"], {"data": {}}, *args, **kwargs)
            return self._add_node(bound["name"], bound["data"])

    def _add_node(self, name: str, data: dict[str, Attribute]) -> Self:
        self.graph.add_node(name)
        self._node_data[name] = data
        return self

    @overload
    def add_edge(self, e: Edge) -> Self: ...

    @overload
    def add_edge(self, src: str, dst: str, data: dict[str, Attribute]) -> Self: ...

    def add_edge(self, *args, **kwargs) -> Self:
        if len(args) + len(kwargs) == 1:
            if len(args) == 1:
                e = args[0]
            else:
                e = kwargs["edge"]
            return self.add_edge(e.src, e.dst, e.data)
        else:
            bound = _bind_args(["src", "dst", "data"], {"data": {}}, *args, **kwargs)
            self.graph.add_edge(bound["src"], bound["dst"])
            self._edge_data[(bound["src"], bound["dst"])] = bound["data"]
            return self

    def add_nodes(self, ns: Iterable[Node]) -> Self:
        for n in ns:
            self.add_node(n)
        return self

    def add_edges(self, es: Iterable[Edge]) -> Self:
        for e in es:
            self.add_edge(e)
        return self

    def successors(self, node: str | Node) -> Mapping[str, N]:
        if not isinstance(node, str):
            node = node.name

        def _successors():
            return self.graph.successors[node]

        return self.get_node_mapping(_successors)

    def predecessors(self, node: str | Node) -> Mapping[str, N]:
        if not isinstance(node, str):
            node = node.name

        def _predecessors():
            return self.graph.predecessors[node]

        return self.get_node_mapping(_predecessors)

    def get_node_mapping(self, keys: Callable[[], Iterable[str]]) -> Mapping[str, N]:
        """Create a read-only mapping of names to Nodes in the order of `keys()`."""
        return _ReadOnlyMappingInOrderAsIterable(keys, self._node_data, self._node_fn)

    def get_edge_mapping(
        self, keys: Callable[[], Iterable[tuple[str, str]]]
    ) -> Mapping[tuple[str, str], E]:
        """Create a read-only mapping of (src, dst) pairs to Edges in the order of `keys()`."""
        return _ReadOnlyMappingInOrderAsIterable(
            keys, self._edge_data, self._construct_edge
        )

    def _construct_edge(
        self, src_dst: tuple[str, str], data: dict[str, Attribute]
    ) -> E:
        src, dst = src_dst
        return self._edge_fn(src, dst, data)

    @read_only_field
    def nodes(self, _: dict[str, Attribute]) -> Mapping[str, N]:
        return self.get_node_mapping(lambda: self.graph.nodes)

    @read_only_field
    def edges(self, _: dict[str, Attribute]) -> Mapping[tuple[str, str], E]:
        return self.get_edge_mapping(self.graph.iter_edges)

    def as_dict(self) -> dict[str, Attribute]:
        data = self.data.copy()
        edges: dict[str, dict[str, Attribute]] = {}
        for (src, dst), d in self._edge_data.items():
            if src in edges:
                edges[src][dst] = d
            else:
                edges[src] = {dst: d}
        data["edges"] = cast(Attribute, edges)

        return data

    def sync_data_with_graph(self) -> Self:
        """Removes nodes/edges from data that are not in the graph, add empty fields for new nodes/edges."""
        nodes_to_remove = set()
        for n in self._nodes:
            if n not in self.graph.nodes:
                nodes_to_remove.add(n)
        for n in nodes_to_remove:
            del self._nodes[n]
        for n in self.graph.nodes:
            if n not in self._nodes:
                self._nodes[n] = {}

        edges_to_remove = set()
        edges_to_keep = set(self.graph.iter_edges())
        for src in self._edges:
            for dst in self._edges[src]:
                if (src, dst) not in edges_to_keep:
                    edges_to_remove.add((src, dst))

        for src, sink in edges_to_remove:
            if src in self._edges:
                del self._edges[src][sink]
            if len(self._edges[src]) == 0:
                del self._edges[src]

        for src, sink in edges_to_keep:
            if src not in self._edges:
                self._edges[src] = {}
            if sink not in self._edges[src]:
                self._edges[src][sink] = {}

        return self

    def load_from_dict(
        self,
        d: dict[str, Any],
    ) -> "Implementation[N, E]":
        """Load attributes, nodes and edges from a dictionary, that was created by `as_dict`.

        Opposed to the constructor, this will add all nodes and edges to the underlying graph.
        :::{important}
        This will override all state of the current object.
        :::
        """
        self.data = d.copy()
        g = self.graph
        self.graph = self.graph.new()
        del g
        for n in self._node_data.keys():
            self.graph.add_node(n)
        for src, dst in self._edge_data.keys():
            self.graph.add_edge(src, dst)

        return self


_K = TypeVar("_K")
_V = TypeVar("_V")
_K2 = TypeVar("_K2")


class _NestedDictToTupleKeyAdapter(MutableMapping[tuple[_K, _K2], _V]):
    def __init__(self, wrapped: dict[_K, dict[_K2, _V]]):
        self.wrapped = wrapped

    def __getitem__(self, k: tuple[_K, _K2]) -> _V:
        return self.wrapped[k[0]][k[1]]

    def __len__(self) -> int:
        acc = 0
        for v in self.wrapped.values():
            acc += len(v)
        return acc

    def __contains__(self, k: object) -> bool:
        if not isinstance(k, tuple) or len(k) != 2:
            return False
        return k[0] in self.wrapped and k[1] in self.wrapped[k[0]]

    def __delitem__(self, key):
        self.wrapped(key[0]).__delitem__(key[1])

    def __iter__(self) -> Iterator[tuple[_K, _K2]]:
        for k0 in self.wrapped:
            for k1 in self.wrapped[k0]:
                yield k0, k1

    def __setitem__(self, k: tuple[_K, _K2], v: _V) -> None:
        if k[0] not in self.wrapped:
            self.wrapped[k[0]] = {}

        self.wrapped[k[0]][k[1]] = v


class _ReadOnlyMappingInOrderAsIterable(Mapping[_K, _V]):
    def __init__(
        self,
        iterable: Callable[[], Iterable[_K]],
        d: Mapping[_K, Any],
        value_constructor: Callable[[_K, Any], _V],
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
        return self._value_constructor(k, self._d[k])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self) == dict(other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self)})"


def _bind_args(keywords: list[str], optionals: dict[str, Any], *args, **kwargs) -> dict:
    bound = dict()
    keys = keywords.copy()
    opts = optionals.copy()
    for key, arg in zip(keys, args):
        bound[key] = arg
    for key, value in kwargs.items():
        if key in bound:
            raise TypeError("failed to bind arguments: multiple values for argument")
        bound[key] = value
    for key, value in opts.items():
        if key not in bound:
            bound[key] = value

    missing_args = set()
    for k in keywords:
        if k not in bound:
            missing_args.add(k)
    if len(missing_args) > 0:
        raise TypeError(f"failed to bind arguments: missing argument {missing_args}")

    return bound
