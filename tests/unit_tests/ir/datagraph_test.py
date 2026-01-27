from collections.abc import Callable
from typing import Any, Protocol, cast

import pytest

import elasticai.creator.ir.datagraph as dgraph
from elasticai.creator.graph.subgraph_matching import find_all_subgraphs
from elasticai.creator.ir import Attribute, AttributeMapping
from elasticai.creator.ir.datagraph import DataGraph
from elasticai.creator.ir.datagraph_impl import (
    DataGraphImpl,
    DefaultDataGraphFactory,
    DefaultIrFactory,
    DefaultNodeEdgeFactory,
    EdgeImpl,
    NodeImpl,
)
from elasticai.creator.ir.datagraph_impl import (
    EdgeImpl as Edge,
)
from elasticai.creator.ir.datagraph_impl import (
    NodeImpl as Node,
)
from elasticai.creator.ir.deserializer import IrDeserializer
from elasticai.creator.ir.factories import (
    IrFactory,
    NodeEdgeFactory,
    StdNodeEdgeFactory,
)
from elasticai.creator.ir.graph import GraphImpl
from elasticai.creator.ir.serializer import IrSerializer


@pytest.fixture
def factory() -> IrFactory[dgraph.Node, dgraph.Edge, dgraph.DataGraph[Node, Edge]]:
    return DefaultIrFactory()


@pytest.fixture
def new_dgraph(factory) -> Callable[[], DataGraphImpl[Node, Edge]]:
    return factory.graph


def test_adding_node_to_datagraph_does_not_modify_original(new_dgraph) -> None:
    g1 = new_dgraph()
    g2 = g1.add_node("a", AttributeMapping(type="N"))

    assert "a" not in g1.successors
    assert "a" in g2.successors


def test_can_add_multiple_nodes(new_dgraph) -> None:
    g1 = new_dgraph().add_nodes(
        ("a", AttributeMapping(type="A")), ("b", AttributeMapping(type="B")), "c"
    )

    assert "a" in g1.successors
    assert "b" in g1.successors
    assert "c" in g1.successors


def test_nodes_compare_equal(new_dgraph) -> None:
    n1 = Node("a", AttributeMapping(type="N", value=10))
    n2 = Node("a", AttributeMapping(type="N", value=10))
    n3 = Node("a", AttributeMapping(type="N", value=20))

    assert n1 == n2
    assert n1 != n3


def test_can_add_and_retrieve_node(new_dgraph) -> None:
    g = new_dgraph().add_node("a", AttributeMapping(type="N"))
    assert Node("a", AttributeMapping(type="N")) == g.nodes["a"]
    assert "N" == g.nodes["a"].attributes["type"]


def test_adding_node_by_string_defaults_to_empty_attribute_map(new_dgraph) -> None:
    g = new_dgraph().add_node("a")
    assert Node("a", AttributeMapping()) == g.nodes["a"]


def test_can_add_edges_to_datagraph(new_dgraph) -> None:
    g = new_dgraph().add_edges(
        ("a", "b", AttributeMapping(type="E1")),
        ("b", "c", AttributeMapping(type="E2")),
    )

    assert {("a", "b"), ("b", "c")} == set(g.edges)


def test_can_remove_node(new_dgraph) -> None:
    g1 = new_dgraph().add_edges(
        ("a", "b", AttributeMapping(type="E1")),
        ("b", "c", AttributeMapping(type="E2")),
    )
    g2 = g1.remove_node("b")

    assert "b" not in g2.nodes
    assert ("a", "b") not in g2.edges
    assert ("b", "c") not in g2.edges


def test_can_remove_edge(new_dgraph) -> None:
    g1 = new_dgraph().add_edges(("a", "b"), ("b", "c"))
    g2 = g1.remove_edge("a", "b")
    assert set(g2.edges.keys()) == {("b", "c")}


def test_can_add_node_attributes_after_adding_edges(new_dgraph) -> None:
    g = (
        new_dgraph()
        .add_edges(("a", "b"))
        .add_nodes(
            Node("a", AttributeMapping(type="N")), Node("b", AttributeMapping(type="N"))
        )
    )
    assert g.nodes["a"].attributes["type"] == "N"
    assert g.nodes["b"].attributes["type"] == "N"


def test_creating_datagraph_from_non_empty_raw_graph_but_missing_node_attributes_can_create_nodes_view():
    raw_g = GraphImpl(lambda: AttributeMapping()).add_nodes("a")
    g = DataGraphImpl(
        factory=DefaultNodeEdgeFactory(),
        attributes=AttributeMapping(),
        graph=raw_g,
        node_attributes=AttributeMapping(),
    )
    assert g.nodes["a"] == DefaultIrFactory().node("a", AttributeMapping())


def test_can_create_dgraph_with_factory(factory):
    g = (
        factory.graph(AttributeMapping(a="b"))
        .add_edges(("x", "y", AttributeMapping(weight=1)))
        .add_nodes(
            ("x", AttributeMapping(type="tx")), ("y", AttributeMapping(type="ty"))
        )
    )
    assert g.attributes == AttributeMapping(a="b")
    assert g.successors["x"]["y"]["weight"] == 1
    assert g.node_attributes["x"]["type"] == "tx"


def test_can_match_datagraph(new_dgraph) -> None:
    g = (
        new_dgraph()
        .add_edges(
            ("a", "b"),
            ("b", "c"),
        )
        .add_nodes(
            Node("a", AttributeMapping(type="A")),
            Node("b", AttributeMapping(type="B")),
            Node("c", AttributeMapping(type="C")),
        )
    )

    pattern = (
        new_dgraph()
        .add_edges(
            ("x", "y"),
        )
        .add_nodes(
            Node("x", AttributeMapping(type="A")), Node("y", AttributeMapping(type="B"))
        )
    )

    def node_constraint(pattern_node: str, graph_node: str) -> bool:
        return (
            pattern.nodes[pattern_node].attributes["type"]
            == g.nodes[graph_node].attributes["type"]
        )

    matches = find_all_subgraphs(
        pattern=pattern, graph=g, node_constraint=node_constraint
    )

    assert len(matches) == 1
    assert matches[0] == {"x": "a", "y": "b"}


def test_create_new_dgraph_from_existing(new_dgraph) -> None:
    class NewDGraph[N: Node, E: Edge](DataGraphImpl[N, E]):
        @property
        def type(self) -> str:
            return self.attributes.get("type", "<undefined>")

    class NewNode(NodeImpl):
        @property
        def shape(self) -> tuple[int, int]:
            return self.attributes.get("shape", (0, 0))

    def convert_data_graph_type[N: Node, E: Edge](
        g: DataGraph[N, E],
    ) -> NewDGraph[NewNode, Edge]:
        return NewDGraph(
            factory=StdNodeEdgeFactory(NewNode, EdgeImpl),
            attributes=g.attributes,
            graph=g.graph,
            node_attributes=g.node_attributes,
        )

    g = new_dgraph().add_node("a", AttributeMapping(type="ta"))
    g2 = convert_data_graph_type(g)
    assert g2.type == "<undefined>"
    assert g2.node_attributes["a"] == AttributeMapping(type="ta")
    assert g2.nodes["a"].shape == (0, 0)


@pytest.fixture
def serialize() -> Callable[[Attribute | Node], Any]:
    s: IrSerializer = IrSerializer()

    def _serialize(attribute: Attribute | Node) -> Any:
        return s.serialize(attribute)

    return _serialize


class TestSerialization:
    def test_can_serialize_primitive_attributes(
        self, serialize: Callable[[Attribute], Any]
    ) -> None:
        a = (1, 2.0, "three", True)
        serialized = serialize(a)
        assert a == serialized

    def test_serializing_list_raises_error(
        self, serialize: Callable[[Attribute], Any]
    ) -> None:
        pytest.raises(TypeError, serialize, [1, 2, 3])

    def test_can_serialize_flat_attribute_mapping(
        self, serialize: Callable[[Attribute], Any]
    ) -> None:
        a = AttributeMapping(x=1, y="two", z=3.0)
        serialized = serialize(a)
        assert serialized == {"x": 1, "y": "two", "z": 3.0}
        assert isinstance(serialized, dict)

    def test_can_serialize_nested_attribute_mapping(
        self, serialize: Callable[[Attribute], Any]
    ) -> None:
        a = AttributeMapping(
            x=1,
            y=AttributeMapping(a="A", b="B"),
        )
        serialized = serialize(a)
        assert serialized == {"x": 1, "y": {"a": "A", "b": "B"}}
        assert isinstance(serialized, dict)
        assert isinstance(serialized["y"], dict)

    def test_hiding_unsupported_type_inside_tuple_raises_error(
        self, serialize: Callable[[Attribute], Any]
    ) -> None:
        a = AttributeMapping(x=1, y=(2, [3, 4]))  # type: ignore
        pytest.raises(TypeError, serialize, a)

    def test_serializing_a_node_means_to_serialize_its_attributes(
        self, serialize: Callable[[Attribute | Node], Any]
    ) -> None:
        n = Node("a", AttributeMapping(type="N", value=10))
        serialized = serialize(n)
        assert serialized == {"type": "N", "value": 10}

    def test_can_serialize_node_subclass(self, serialize) -> None:
        class MyNode(Node):
            def __init__(self, name: str, data: AttributeMapping) -> None:
                super().__init__(name, AttributeMapping(extra="extra") | data)

            @property
            def extra(self) -> str:
                return self.attributes["extra"]

        n = MyNode("a", AttributeMapping(type="N"))
        serialized = serialize(n)
        assert serialized == {"extra": "extra", "type": "N"}

    def test_can_serialize_datagraph(self, serialize, factory) -> None:
        g: dgraph.DataGraph[Node, Edge] = factory.graph(
            attributes=AttributeMapping(version=1)
        )
        g = g.add_edges(
            ("a", "b"),
        )
        g = g.add_nodes(
            factory.node("a", AttributeMapping(type="A")),
            factory.node("b", AttributeMapping(type="B")),
        )

        serialized = serialize(g)
        assert serialized == {
            "nodes": {
                "a": {"type": "A"},
                "b": {"type": "B"},
            },
            "edges": {"a": {"b": {}}, "b": {}},
            "attributes": {"version": 1},
        }

    def test_deserialization(self, serialize, factory) -> None:
        g: dgraph.DataGraph[Node, Edge] = factory.graph(
            attributes=AttributeMapping(version=1)
        )
        g = g.add_edges(
            ("a", "b"),
        )
        g = g.add_nodes(
            ("a", AttributeMapping(type="A")),
            ("b", AttributeMapping(type="B")),
        )

        serialized = serialize(g)
        deserializer = IrDeserializer(DefaultIrFactory())
        deserialized = deserializer.deserialize_graph(serialized)
        print(deserialized.attributes)
        print(g.attributes)
        print(g.successors)
        print(deserialized.successors)
        print(serialized)

        assert g == deserialized


def test_dynamically_extend_nodes_and_graphs_in_custom_factory() -> None:
    class MyNode(dgraph.Node, Protocol):
        @property
        def implementation(self) -> str: ...

    class MyGraph(dgraph.DataGraph[MyNode, dgraph.Edge], Protocol):
        @property
        def name(self) -> str: ...

    class MyIrFactory(IrFactory):
        def __init__(self):
            self._node_edge = DefaultNodeEdgeFactory()
            self._graph = DefaultDataGraphFactory(self)

        def node(
            self, name: str, attributes: AttributeMapping = AttributeMapping()
        ) -> MyNode:
            n = self._node_edge.node(name, attributes)

            class _Node(n.__class__):  # type: ignore
                @property
                def implementation(self) -> str:
                    return self.attributes.get("implementation", "<none>")

            n.__class__ = _Node
            return n

        def edge(
            self,
            src: str,
            dst: str,
            attributes: AttributeMapping = AttributeMapping(),
        ) -> Edge:
            return self._node_edge.edge(src, dst, attributes)

        def graph(self, attributes: AttributeMapping = AttributeMapping()) -> MyGraph:
            g = self._graph.graph(attributes)

            class _Graph(g.__class__):  # type: ignore
                @property
                def name(self) -> str:
                    return self.attributes.get("name", "<undefined>")

            g.__class__ = _Graph
            return g

    factory = MyIrFactory()
    n = factory.node("x")
    g = factory.graph()
    assert n.implementation == "<none>"
    assert g.name == "<undefined>"


def test_extend_node_and_graph_statically() -> None:
    class MyNode(Node):
        @property
        def implementation(self) -> str:
            return cast(str, self.attributes.get("implementation", "<none>"))

    class MyGraph[N: Node, E: Edge](DataGraphImpl[N, E]):
        @property
        def name(self) -> str:
            return cast(str, self.attributes.get("name", "<undefined>"))

    class MyFactory(IrFactory):
        def node(
            self, name: str, attributes: AttributeMapping = AttributeMapping()
        ) -> MyNode:
            return MyNode(name, attributes)

        def edge(
            self, src: str, dst: str, attributes: AttributeMapping = AttributeMapping()
        ) -> Edge:
            return Edge(src, dst, attributes)

        def graph(
            self, attributes: AttributeMapping = AttributeMapping()
        ) -> MyGraph[MyNode, Edge]:
            return MyGraph(
                factory=self,
                attributes=attributes,
                graph=GraphImpl(
                    lambda: AttributeMapping(),
                ),
                node_attributes=AttributeMapping(),
            )

    f = MyFactory()
    n = f.node("x")
    g = f.graph()
    assert n.implementation == "<none>"
    assert g.name == "<undefined>"


def test_wrap_existing_data_into_new_graph_with_new_node_type(factory) -> None:
    class MyNode(Node):
        @property
        def implementation(self) -> str:
            return cast(str, self.attributes.get("implementation", "<none>"))

    class MyGraph[N: Node, E: Edge](DataGraphImpl[N, E]):
        @property
        def name(self) -> str:
            return cast(str, self.attributes.get("name", "<undefined>"))

    class MyNodeEdgeFactory(NodeEdgeFactory[MyNode, Edge]):
        def node(
            self, name: str, attributes: AttributeMapping = AttributeMapping()
        ) -> MyNode:
            return MyNode(name, attributes)

        def edge(
            self, src: str, dst: str, attributes: AttributeMapping = AttributeMapping()
        ) -> Edge:
            return Edge(src, dst, attributes)

    def to_my_graph_type(g: dgraph.DataGraph) -> MyGraph[MyNode, Edge]:
        return MyGraph(
            factory=MyNodeEdgeFactory(),
            attributes=g.attributes,
            graph=g.graph,
            node_attributes=g.node_attributes,
        )

    g1: DataGraphImpl[Node, Edge] = factory.graph()
    g1 = g1.add_edges(("a", "b"), ("c", "d"))
    g2: MyGraph[MyNode, Edge] = to_my_graph_type(g1).add_node("e")
    assert g2.name == "<undefined>"
    assert g2.nodes["a"].implementation == "<none>"
    assert g2.nodes["e"].implementation == "<none>"
