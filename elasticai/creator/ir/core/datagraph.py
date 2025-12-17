from collections.abc import (
    Collection,
    Iterator,
    Mapping,
)
from typing import Self, overload

from .attribute import AttributeMapping
from .factories import NodeEdgeFactory
from .graph import Graph, GraphImpl


class Node:
    def __init__(self, name: str, attributes: AttributeMapping) -> None:
        self._attributes = attributes
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def attributes(self) -> AttributeMapping:
        return self._attributes

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"Node({self._name}, {repr(self._attributes)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self._name == other._name and self._attributes == other._attributes


class Edge:
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
            self._src == other._src
            and self._dst == other._dst
            and self._attributes == other._attributes
        )


class DefaultNodeEdgeFactory(NodeEdgeFactory[Node, Edge]):
    def node(self, name: str, attributes: AttributeMapping) -> Node:
        return Node(name, attributes)

    def edge(self, src: str, dst: str, attributes: AttributeMapping) -> Edge:
        return Edge(src, dst, attributes)


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
        attributes = self._attributes[key]
        return self._factory.node(key, attributes)

    def __iter__(self) -> Iterator[str]:
        return iter(self._successors)

    def __repr__(self) -> str:
        return f"NodeView({repr(self._attributes)})"

    def __len__(self) -> int:
        return len(self._successors)


class DataGraphImpl[N: Node, E: Edge](Graph[str, AttributeMapping]):
    def __init__(
        self,
        factory: NodeEdgeFactory[N, E] = DefaultNodeEdgeFactory(),  # type: ignore # while we do not have default type args
        attributes: AttributeMapping = AttributeMapping(),
        graph: Graph[str, AttributeMapping] = GraphImpl(lambda: AttributeMapping()),
        node_attributes: AttributeMapping = AttributeMapping(),
    ) -> None:
        self._factory: NodeEdgeFactory[N, E] = factory
        self._graph: Graph[str, AttributeMapping] = graph
        self._attributes = attributes
        self._nodes = node_attributes

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
    def edges(self) -> EdgeView[E]:
        return EdgeView(self._graph.successors, self._factory)

    @property
    def nodes(self) -> NodeView[N]:
        return NodeView(
            self._graph.successors.keys(),
            self._nodes,
            self._factory,
        )

    @overload
    def add_edge(self, edge: E, /) -> Self: ...

    @overload
    def add_edge(
        self, src: str, dst: str, attributes: AttributeMapping | None = None, /
    ) -> Self: ...

    def add_edge(
        self,
        src: str | E,
        dst: str | None = None,
        attributes: AttributeMapping | None = None,
        /,
    ) -> Self:
        """Updates edge in case it exists already. Possibly already existing nodes remain unchanged."""
        return self.add_edges(self._dispatch_edge_args((src, dst, attributes)))

    def add_edges(
        self, *edges: E | tuple[str, str, AttributeMapping] | tuple[str, str]
    ) -> Self:
        """Updates edges in case they exist already. Possibly already existing nodes remain unchanged."""
        dispatched = tuple(map(self._dispatch_edge_args, edges))
        new_graph = self._graph.add_edges(*dispatched)
        new_nodes = {}
        for src, dst, _ in dispatched:
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

    def _dispatch_edge_args(
        self,
        arg: E
        | tuple[str, str]
        | tuple[str, str, AttributeMapping]
        | tuple[E, None]
        | tuple[str | E, str | None, AttributeMapping | None],
    ) -> tuple[str, str, AttributeMapping]:
        if isinstance(arg, Edge):
            return arg.src, arg.dst, arg.attributes
        elif isinstance(arg, tuple):
            if isinstance(arg[0], Edge):
                return self._dispatch_edge_args(arg[0])
            elif isinstance(arg[0], str) and isinstance(arg[1], str):
                src, dst, *attributes = arg
                if attributes is None or attributes == []:
                    attributes = [AttributeMapping()]
                e = self._factory.edge(src, dst, attributes[0])  # type: ignore
                return e.src, e.dst, e.attributes
            else:
                raise TypeError(f"Invalid edge arguments {arg}.")
        else:
            raise TypeError(f"Invalid edge argument {arg}.")

    @overload
    def add_node(self, node: N, /) -> Self: ...

    @overload
    def add_node(
        self, name: str, attributes: AttributeMapping | None = None, /
    ) -> Self: ...

    def add_node(
        self, name: str | N, attributes: AttributeMapping | None = None, /
    ) -> Self:
        """Updates node in case it exists already."""
        return self.add_nodes(self._dispatch_node_args((name, attributes)))

    def add_nodes(
        self,
        *nodes: N | tuple[str, AttributeMapping] | str,
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
        return type(self)(
            attributes=self._attributes,
            factory=self._factory,
            graph=new_graph,
            node_attributes=new_nodes,
        )

    def _dispatch_node_args(
        self,
        arg: N
        | str
        | tuple[str, AttributeMapping]
        | tuple[N, AttributeMapping]
        | tuple[str | N, AttributeMapping | None],
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
        return type(self)(
            attributes=self._attributes,
            factory=self._factory,
            graph=new_graph,
            node_attributes=new_nodes,
        )

    def remove_edge(self, src: str, dst: str) -> Self:
        """Will not remove nodes, even if they become isolated."""
        new_graph = self._graph.remove_edge(src, dst)
        return type(self)(
            attributes=self._attributes,
            factory=self._factory,
            graph=new_graph,
            node_attributes=self._nodes,
        )

    def with_attributes(self, attributes: AttributeMapping) -> Self:
        return type(self)(
            attributes=attributes,
            factory=self._factory,
            graph=self._graph,
            node_attributes=self._nodes,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataGraphImpl):
            return False
        return (
            self._attributes == other._attributes
            and self._graph == other._graph
            and self._nodes == other._nodes
        )

    def new(self) -> Self:
        return type(self)(
            attributes=AttributeMapping(),
            factory=self._factory,
            graph=GraphImpl(lambda: AttributeMapping()),
            node_attributes=AttributeMapping(),
        )


class DefaultIrFactory:
    def node(self, name: str, attributes: AttributeMapping) -> Node:
        return Node(name, attributes)

    def edge(self, src: str, dst: str, attributes: AttributeMapping) -> Edge:
        return Edge(src, dst, attributes)

    def graph(self, attributes: AttributeMapping) -> DataGraphImpl[Node, Edge]:
        return DataGraphImpl(attributes=attributes, factory=self)
