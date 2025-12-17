from collections.abc import Callable
from typing import Any

import pytest

from elasticai.creator.graph.subgraph_matching import find_all_subgraphs
from elasticai.creator.ir.core.attribute import Attribute, AttributeMapping
from elasticai.creator.ir.core.datagraph import (
    DataGraphImpl,
    DefaultIrFactory,
    GraphImpl,
    Node,
)
from elasticai.creator.ir.core.deserializer import IrDeserializer
from elasticai.creator.ir.core.graph import AdjacencyMap
from elasticai.creator.ir.core.serializer import IrSerializer


def test_nested_attribute_map_is_immutable() -> None:
    a = AttributeMapping(x=1, y=2)
    b = AttributeMapping(a=a, b=(3, 4))
    c = b.new_with(a=a.new_with(x=10))
    assert a["y"] == 2
    assert b["a"]["x"] == 1
    assert c["a"]["x"] == 10


def test_nested_mapping_can_merge_with_other_nested_mapping() -> None:
    a = AdjacencyMap({"x": {"y": 1}})
    b = AdjacencyMap({"x": {"z": 2}})
    c = a.join(b)
    assert c["x"] == {"y": 1, "z": 2}


def test_nested_mapping_holding_attribute_map_is_immutable() -> None:
    a = AdjacencyMap({"x": {"y": AttributeMapping(a=1)}})
    b = AdjacencyMap({"x": {"y": AttributeMapping(b=2)}})
    c = a.join(b)
    assert a["x"]["y"] == {"a": 1}
    assert c["x"]["y"] == {"b": 2}


def test_can_create_new_nested_mapping_by_merging_with_dict() -> None:
    a: AdjacencyMap[str, int] = AdjacencyMap({})
    b = a.join({"x": {"y": 1}})
    assert b == {"x": {"y": 1}}


def adding_node_does_not_modify_original_graph() -> None:
    g1: GraphImpl[str, AttributeMapping] = GraphImpl(lambda: AttributeMapping())
    g2 = g1.add_node("a")

    assert "a" not in g1.successors
    assert "a" in g2.successors


def test_can_add_edges_without_attributes_to_graph() -> None:
    g1: GraphImpl[str, AttributeMapping] = GraphImpl(lambda: AttributeMapping())
    g2 = g1.add_edges(("a", "b"))

    assert AttributeMapping() == g2.successors["a"]["b"]


def adding_edge_does_not_modify_original_graph() -> None:
    g1: GraphImpl[str, AttributeMapping] = GraphImpl(lambda: AttributeMapping())
    g2 = g1.add_edge("a", "b", AttributeMapping(type="E"))

    def iter_edges[N, E](graph):
        for src in graph.successors:
            for dst, attributes in graph.successors[src].items():
                yield src, dst, attributes

    assert ("a", "b") not in list(iter_edges(g1))
    assert ("a", "b") in list(iter_edges(g2))


def can_add_multiple_edges_at_once() -> None:
    g1: GraphImpl[str, AttributeMapping] = GraphImpl(lambda: AttributeMapping())
    g2 = g1.add_edges(
        ("a", "b", AttributeMapping(type="A")),
        ("b", "c", AttributeMapping(type="B")),
    )

    g2_edges = set()
    for src in g2.successors:
        for dst in g2.successors[src]:
            g2_edges.add((src, dst))

    assert ("a", "b") in g2_edges
    assert ("b", "c") in g2_edges


def test_adding_node_to_datagraph_does_not_modify_original() -> None:
    g1 = DataGraphImpl()
    g2 = g1.add_node("a", AttributeMapping(type="N"))

    assert "a" not in g1.successors
    assert "a" in g2.successors


def test_can_add_multiple_nodes() -> None:
    g1 = DataGraphImpl().add_nodes(
        ("a", AttributeMapping(type="A")), ("b", AttributeMapping(type="B")), "c"
    )

    assert "a" in g1.successors
    assert "b" in g1.successors
    assert "c" in g1.successors


def test_nodes_compare_equal() -> None:
    n1 = Node("a", AttributeMapping(type="N", value=10))
    n2 = Node("a", AttributeMapping(type="N", value=10))
    n3 = Node("a", AttributeMapping(type="N", value=20))

    assert n1 == n2
    assert n1 != n3


def test_can_add_and_retrieve_node() -> None:
    g = DataGraphImpl().add_node("a", AttributeMapping(type="N"))
    assert Node("a", AttributeMapping(type="N")) == g.nodes["a"]
    assert "N" == g.nodes["a"].attributes["type"]


def test_adding_node_by_string_defaults_to_empty_attribute_map() -> None:
    g = DataGraphImpl().add_node("a")
    assert Node("a", AttributeMapping()) == g.nodes["a"]


def test_can_add_edges_to_datagraph() -> None:
    g = DataGraphImpl().add_edges(
        ("a", "b", AttributeMapping(type="E1")),
        ("b", "c", AttributeMapping(type="E2")),
    )

    assert {("a", "b"), ("b", "c")} == set(g.edges)


def test_updating_nested_mapping_with_empty_submaps_adds_empty_submaps() -> None:
    nm1 = AdjacencyMap({"a": {"x": 1}})
    nm2 = nm1.join({"b": {}})

    assert "b" in nm2
    assert nm2["b"] == {}


def test_adding_edge_adds_nodes_to_predecessors_and_successors() -> None:
    g: GraphImpl[str, AttributeMapping] = GraphImpl(lambda: AttributeMapping())
    g = g.add_edge("a", "b", AttributeMapping(type="E"))

    assert "a" in g.successors
    assert "a" in g.predecessors


def test_can_drop_key_pair_from_nested_mapping() -> None:
    nm1 = AdjacencyMap({"a": {"x": 1, "y": 2}, "b": {"z": 3}})
    nm2 = nm1.drop("a", "y")

    assert nm2["a"] == {"x": 1}


def test_can_drop_key_from_nested_mapping() -> None:
    nm1 = AdjacencyMap({"a": {"x": 1, "y": 2}, "x": {"z": 3}})
    nm2 = nm1.drop("x")
    assert "x" not in nm2["a"]
    assert "x" not in nm2


def test_can_remove_node() -> None:
    g1 = DataGraphImpl().add_edges(
        ("a", "b", AttributeMapping(type="E1")),
        ("b", "c", AttributeMapping(type="E2")),
    )
    g2 = g1.remove_node("b")

    assert "b" not in g2.nodes
    assert ("a", "b") not in g2.edges
    assert ("b", "c") not in g2.edges


def test_can_add_node_attributes_after_adding_edges() -> None:
    g = (
        DataGraphImpl()
        .add_edges(("a", "b"))
        .add_nodes(
            Node("a", AttributeMapping(type="N")), Node("b", AttributeMapping(type="N"))
        )
    )
    assert g.nodes["a"].attributes["type"] == "N"
    assert g.nodes["b"].attributes["type"] == "N"


def test_can_match_datagraph() -> None:
    g = (
        DataGraphImpl()
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
        DataGraphImpl()
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


@pytest.fixture
def serialize() -> Callable[[Attribute | Node], Any]:
    s = IrSerializer()

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

    def test_can_serialize_datagraph(self, serialize) -> None:
        g = (
            DataGraphImpl(attributes=AttributeMapping(version=1))
            .add_edges(
                ("a", "b"),
            )
            .add_nodes(
                Node("a", AttributeMapping(type="A")),
                Node("b", AttributeMapping(type="B")),
            )
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

    def test_deserialization(self, serialize) -> None:
        g = (
            DataGraphImpl(attributes=AttributeMapping(version=1))
            .add_edges(
                ("a", "b"),
            )
            .add_nodes(
                Node("a", AttributeMapping(type="A")),
                Node("b", AttributeMapping(type="B")),
            )
        )

        serialized = serialize(g)
        deserializer = IrDeserializer(DefaultIrFactory())
        deserialized = deserializer.deserialize_graph(serialized)

        assert g == deserialized
