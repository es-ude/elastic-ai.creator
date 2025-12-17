from collections.abc import Callable

import pytest

import elasticai.creator.ir.datagraph as dgraph
from elasticai.creator.graph.subgraph_matching import find_all_subgraphs
from elasticai.creator.ir.attribute import AttributeMapping
from elasticai.creator.ir.datagraph import DataGraph
from elasticai.creator.ir.datagraph_impl import (
    DataGraphImpl,
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
from elasticai.creator.ir.factories import (
    IrFactory,
    StdNodeEdgeFactory,
)
from elasticai.creator.ir.graph import GraphImpl


@pytest.fixture
def factory() -> IrFactory[dgraph.Node, dgraph.Edge, DataGraph[Node, Edge]]:
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
