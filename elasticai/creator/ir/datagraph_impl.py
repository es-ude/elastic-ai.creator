from collections.abc import (
    Collection,
    Iterator,
    Mapping,
)
from typing import Self

from .attribute import AttributeMapping
from .datagraph import DataGraph, Edge, Node, ReadOnlyDataGraph
from .factories import (
    NodeEdgeFactory,
    StdDataGraphFactory,
    StdIrFactory,
)
from .graph import Graph, GraphImpl


class NodeImpl(Node):
    __slots__ = ("_name", "_attributes")

    def __init__(self, name: str, attributes: AttributeMapping) -> None:
        self._attributes = attributes
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        if "type" not in self._attributes:
            return "<undefined>"
        return self._attributes["type"]

    @property
    def attributes(self) -> AttributeMapping:
        return self._attributes

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"Node({self._name}, {repr(self._attributes)})"

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Node):
            return False
        return self._name == o.name and self._attributes == o.attributes


class EdgeImpl(Edge):
    __slots__ = ("_src", "_dst", "_attributes")

    def __init__(self, src: str, dst: str, attributes: AttributeMapping) -> None:
        self._attributes = attributes
        self._src = src
        self._dst = dst

    @property
    def attributes(self) -> AttributeMapping:
        return self._attributes

    @property
    def src(self) -> str:
        return self._src

    @property
    def dst(self) -> str:
        return self._dst

    def __hash__(self) -> int:
        return hash((self._src, self._dst))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return (
            self._src == other.src
            and self._dst == other.dst
            and self._attributes == other.attributes
        )


class DefaultNodeEdgeFactory(NodeEdgeFactory[Node, Edge]):
    def node(
        self, name: str, attributes: AttributeMapping = AttributeMapping()
    ) -> Node:
        return NodeImpl(name, attributes)

    def edge(
        self, src: str, dst: str, attributes: AttributeMapping = AttributeMapping()
    ) -> Edge:
        return EdgeImpl(src, dst, attributes)


class EdgeView[E: Edge](Mapping[tuple[str, str], E]):
    def __init__(
        self,
        successors: Mapping[str, Mapping[str, AttributeMapping]],
        factory: NodeEdgeFactory[Node, E],
    ) -> None:
        self._successors = successors
        self._factory = factory

    def __getitem__(self, key: tuple[str, str]) -> E:
        src, dst = key
        return self._factory.edge(src, dst, self._successors[src][dst])

    def __iter__(self) -> Iterator[tuple[str, str]]:
        for src in self._successors:
            for dst in self._successors[src]:
                yield (src, dst)

    def __len__(self) -> int:
        length = 0
        for src in self._successors:
            length += len(self._successors[src])
        return length


class NodeView[N: Node](Mapping[str, N]):
    def __init__(
        self,
        successors: Collection[str],
        attributes: AttributeMapping,
        factory: NodeEdgeFactory[N, Edge],
    ) -> None:
        self._successors = successors
        self._factory = factory
        self._attributes = attributes

    def __getitem__(self, key: str) -> N:
        if key not in self._successors:
            raise KeyError(key)
        if key not in self._attributes:
            return self._factory.node(key)
        else:
            return self._factory.node(key, self._attributes[key])

    def __contains__(self, object) -> bool:
        return object in self._successors

    def __iter__(self) -> Iterator[str]:
        return iter(self._successors)

    def __repr__(self) -> str:
        return f"NodeView({repr(self._attributes)})"

    def __len__(self) -> int:
        return len(self._successors)


class DataGraphImpl[N: Node, E: Edge](DataGraph[N, E]):
    """Keep constructor signature when subclassing!"""

    __slots__ = ("_factory", "_graph", "_attributes", "_nodes")

    def __init__(
        self,
        /,
        factory: NodeEdgeFactory[N, E],
        attributes: AttributeMapping,
        graph: Graph[str, AttributeMapping],
        node_attributes: AttributeMapping,
    ) -> None:
        self._factory: NodeEdgeFactory[N, E] = factory
        self._graph: Graph[str, AttributeMapping] = graph
        self._attributes = attributes
        self._nodes = node_attributes

    @property
    def node_attributes(self) -> AttributeMapping:
        return self._nodes

    @property
    def graph(self) -> Graph[str, AttributeMapping]:
        return self._graph

    def new_from_read_only_data_graph(self, g: ReadOnlyDataGraph) -> Self:
        return self.new(
            attributes=g.attributes,
            node_attributes=g.node_attributes,
            graph=g.graph,
        )

    @property
    def attributes(self) -> AttributeMapping:
        return self._attributes

    @property
    def successors(self) -> Mapping[str, Mapping[str, AttributeMapping]]:
        return self._graph.successors

    @property
    def predecessors(self) -> Mapping[str, Mapping[str, AttributeMapping]]:
        return self._graph.predecessors

    @property
    def edges(self) -> Mapping[tuple[str, str], E]:
        return EdgeView(self._graph.successors, self._factory)

    @property
    def nodes(self) -> Mapping[str, N]:
        return NodeView(
            self._graph.successors.keys(),
            self._nodes,
            self._factory,
        )

    def add_edge(
        self,
        src: str | Edge,
        dst: str | None = None,
        attributes: AttributeMapping | None = None,
        /,
    ) -> Self:
        """Updates edge in case it exists already. Possibly already existing nodes remain unchanged."""
        if isinstance(src, str) and isinstance(dst, str):
            if attributes is None:
                args = self._unify_edge_args((src, dst))
            else:
                args = self._unify_edge_args((src, dst, attributes))
        elif isinstance(src, Edge) and (dst, attributes) == (None, None):
            args = self._unify_edge_args((src.src, src.dst, src.attributes))
        else:
            raise TypeError()
        return self.add_edges(args)

    def add_edges(
        self, *edges: Edge | tuple[str, str, AttributeMapping] | tuple[str, str]
    ) -> Self:
        """Updates edges in case they exist already. Possibly already existing nodes remain unchanged."""
        dispatched = tuple(map(self._unify_edge_args, edges))
        new_graph = self._graph.add_edges(*dispatched)
        new_nodes = {}
        for src, dst, *_ in dispatched:
            if src not in self._nodes:
                new_nodes[src] = self._factory.node(src, AttributeMapping()).attributes
            if dst not in self._nodes:
                new_nodes[dst] = self._factory.node(dst, AttributeMapping()).attributes
        new_nodes_attributes = self._nodes | new_nodes
        return type(self)(
            attributes=self._attributes,
            factory=self._factory,
            graph=new_graph,
            node_attributes=new_nodes_attributes,
        )

    def _unify_edge_args(
        self,
        arg: Edge
        | tuple[str, str]
        | tuple[str, str, AttributeMapping]
        | tuple[Edge, None]
        | tuple[Edge, None, None]
        | tuple[str, str, None],
    ) -> tuple[str, str, AttributeMapping] | tuple[str, str]:
        if isinstance(
            arg, Edge
        ):  # check only to silence pyrefly until version > v0.47.0
            return arg.src, arg.dst, arg.attributes
        match arg:
            case (
                (Edge() as edge) | (Edge() as edge, None) | (Edge() as edge, None, None)  # type: ignore
            ):
                # see https://github.com/python/mypy/issues/19995 on the type ignore above
                return edge.src, edge.dst, edge.attributes
            case (str() as src, str() as dst) | (str() as src, str() as dst, None):
                return src, dst
            case (str() as src, str() as dst, AttributeMapping() as attr):
                return src, dst, attr
            case _:
                raise TypeError(f"Invalid edge arguments {arg}.")

    def add_node(
        self, name: str | Node, attributes: AttributeMapping | None = None, /
    ) -> Self:
        """Updates node in case it exists already."""
        return self.add_nodes(self._dispatch_node_args((name, attributes)))

    def add_nodes(
        self,
        *nodes: Node | tuple[str, AttributeMapping] | str,
    ) -> Self:
        """Updates nodes in case they exist already."""
        new_graph = self._graph
        new_nodes_attributes: dict[str, AttributeMapping] = {}

        for node in nodes:
            name, attributes = self._dispatch_node_args(node)
            new_nodes_attributes[name] = attributes
        new_graph = self._graph.add_nodes(
            *(name for name, _ in new_nodes_attributes.items())
        )
        new_nodes = self._nodes | new_nodes_attributes
        return self.new(
            attributes=self._attributes,
            graph=new_graph,
            node_attributes=new_nodes,
        )

    def _dispatch_node_args(
        self,
        arg: Node
        | str
        | tuple[str, AttributeMapping]
        | tuple[Node, AttributeMapping]
        | tuple[str | Node, AttributeMapping | None],
    ) -> tuple[str, AttributeMapping]:
        if isinstance(arg, Node):
            attributes = arg.attributes
            name = arg.name
        elif isinstance(arg, str):
            attributes = None
            name = arg
        elif isinstance(arg, tuple):
            if isinstance(arg[0], Node):
                name = arg[0].name
                attributes = arg[0].attributes
            elif isinstance(arg[0], str):
                name = arg[0]
                attributes = arg[1]
            else:
                raise TypeError(f"Invalid node arguments {arg}.")
        else:
            raise TypeError(f"Invalid node argument {arg}.")
        if attributes is None:
            attributes = self._factory.node(name, AttributeMapping()).attributes
        return name, attributes

    def remove_node(self, node: str, /) -> Self:
        """Will remove node and all connected edges."""
        new_graph = self._graph.remove_node(node)
        new_nodes = self._nodes.drop(node)
        return self.new(
            attributes=self._attributes,
            graph=new_graph,
            node_attributes=new_nodes,
        )

    def remove_edge(self, src: str, dst: str) -> Self:
        """Will not remove nodes, even if they become isolated."""
        new_graph = self._graph.remove_edge(src, dst)
        return self.new(
            attributes=self._attributes,
            graph=new_graph,
            node_attributes=self._nodes,
        )

    def with_attributes(self, attributes: AttributeMapping) -> Self:
        return self.new(
            attributes=attributes,
            graph=self._graph,
            node_attributes=self._nodes,
        )

    @property
    def factory(self) -> NodeEdgeFactory[N, E]:
        return self._factory

    def new(
        self,
        /,
        attributes: AttributeMapping,
        graph: Graph[str, AttributeMapping],
        node_attributes: AttributeMapping,
    ) -> Self:
        return type(self)(
            factory=self.factory,
            attributes=attributes,
            graph=graph,
            node_attributes=node_attributes,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataGraphImpl):
            return False
        return (
            self._attributes == other._attributes
            and self._graph == other._graph
            and self._nodes == other._nodes
        )


def _new_graph[N: Node, E: Edge](
    factory: NodeEdgeFactory[N, E], attributes
) -> DataGraph[N, E]:
    return DataGraphImpl(
        factory,
        attributes,
        GraphImpl(lambda: AttributeMapping()),
        AttributeMapping(),
    )


class DefaultDataGraphFactory[N: Node, E: Edge](StdDataGraphFactory[N, E, DataGraph]):
    def __init__(self, node_edge_factory: NodeEdgeFactory[N, E]) -> None:
        super().__init__(node_edge_factory, _new_graph)


class DefaultIrFactory(StdIrFactory[Node, Edge, DataGraph[Node, Edge]]):
    def __init__(self) -> None:
        super().__init__(NodeImpl, EdgeImpl, _new_graph)
