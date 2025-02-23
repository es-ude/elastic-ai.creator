from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import (
    Any,
    Generic,
    ParamSpec,
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


class Implementation(IrData, Generic[N, E], create_init=False):
    __slots__ = (
        "data",
        "graph",
        "_node_data",
        "_edge_data",
        "_node_fn",
        "_edge_fn",
    )

    name: str
    type: str

    @overload
    def __init__(
        self: "Implementation[N, E]",
        *,
        graph: Graph[str],
        edge_fn: Callable[[dict[str, Attribute]], E],
        node_fn: Callable[[dict[str, Attribute]], N],
        data: dict[str, Attribute] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Implementation[N, Edge]",
        *,
        graph: Graph,
        node_fn: Callable[[dict[str, Attribute]], N],
        data: dict[str, Attribute] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Implementation[Node, E]",
        *,
        graph: Graph[str],
        edge_fn: Callable[[dict[str, Attribute]], E],
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
        node_fn: Callable[[dict[str, Attribute]], Any] = Node,
        edge_fn: Callable[[dict[str, Attribute]], Any] = Edge,
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

        for src, sink in original_impl.edges:
            if fulfills_constraint(original_impl.nodes[src]) and fulfills_constraint(original_impl.nodes[sink]):
                subgraph.add_edge(src, sink)

        new_impl = Implementation(graph=subgraph, data=original_impl.data)

        for node in new_impl.nodes.values():  # now we can iterate over the nodes of the subgraph
            do_something(node)
        ```
        """
        if data is None:
            data = {}
        if "nodes" not in data:
            data["nodes"] = {}
        if "edges" not in data:
            data["edges"] = {}
        super().__init__(data)

        self.graph = graph
        self._node_data: dict[str, Attribute] = cast(
            dict[str, Attribute], data["nodes"]
        )
        self._edge_data: dict[tuple[str, str], Attribute] = cast(
            dict[tuple[str, str], Attribute], data["edges"]
        )
        self._node_fn = node_fn
        self._edge_fn = edge_fn

    @overload
    def add_node(self, n: Node) -> Self: ...

    @overload
    def add_node(self, n: dict[str, Attribute]) -> Self: ...

    @overload
    def add_node(self, *, name: str, type: str, **attributes: Attribute) -> Self: ...

    def add_node(self, *args, **kwargs) -> Self:
        if len(args) == 1 and not kwargs:
            if isinstance(args[0], Node):
                return self.add_node(args[0].data)
            elif isinstance(args[0], dict):
                return self._add_node(args[0])
        elif len(args) == 0 and "name" in kwargs and "type" in kwargs:
            return self.add_node(kwargs)
        else:
            raise TypeError(
                "invalid arguments, expected either a Node or a dict or keyword arguments name, type, and more optional attributes"
            )
        return self

    def _add_node(self, n: dict[str, Attribute]) -> Self:
        self.graph.add_node(cast(str, n["name"]))
        self._node_data[cast(str, n["name"])] = n
        return self

    @overload
    def add_edge(self, e: Edge) -> Self: ...

    @overload
    def add_edge(self, e: dict) -> Self: ...

    @overload
    def add_edge(self, *, src: str, sink: str, **attributes: Attribute) -> Self: ...

    def add_edge(self, *args, **kwargs) -> Self:
        if len(args) == 1 and not kwargs:
            if isinstance(args[0], Edge):
                return self.add_edge(args[0].data)
            elif isinstance(args[0], dict):
                return self._add_edge(args[0])
        elif len(args) == 0 and "src" in kwargs and "sink" in kwargs:
            return self.add_edge(kwargs)
        raise TypeError(
            "invalid arguments, expected either an Edge or a dict or keyword arguments src, sink, and more optional attributes"
        )

    def _add_edge(self, e: dict) -> Self:
        self.graph.add_edge(e["src"], e["sink"])
        self._edge_data[(e["src"], e["sink"])] = e
        return self

    def add_nodes(self, ns: Iterable[Node | dict[str, Attribute]]) -> Self:
        for n in ns:
            self.add_node(n)
        return self

    def add_edges(self, es: Iterable[Edge | dict[str, Attribute]]) -> Self:
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
        """Create a read-only mapping of (src, sink) pairs to Edges in the order of `keys()`."""
        return _ReadOnlyMappingInOrderAsIterable(keys, self._edge_data, self._edge_fn)

    @read_only_field
    def nodes(self, _: dict[str, Attribute]) -> Mapping[str, N]:
        return self.get_node_mapping(lambda: self.graph.nodes)

    @read_only_field
    def edges(self, _: dict[str, Attribute]) -> Mapping[tuple[str, str], E]:
        return self.get_edge_mapping(self.graph.iter_edges)

    def as_dict(self) -> dict[str, Attribute]:
        data = self.data.copy()
        data["nodes"] = cast(
            Attribute, list(n.data for n in self.nodes.values())
        )  # seems like mypy can't handle the nested recursive type hint list[dict[str, Attribute]]
        data["edges"] = cast(Attribute, list(e.data for e in self.edges.values()))
        return data

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
        data = d.copy()
        nodes = {node["name"]: node for node in data["nodes"]}
        edges = {(edge["src"], edge["sink"]): edge for edge in data["edges"]}
        data["nodes"] = nodes
        data["edges"] = edges
        for node in nodes.values():
            self.add_node(node)
        for edge in edges.values():
            self.add_edge(edge)
        self.data = data
        return self


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
