import json

import pytest

from elasticai.creator.graph import BaseGraph as _Graph
from elasticai.creator.ir import Attribute, Edge, Implementation, Node, edge, node


class Graph(Implementation[Node, Edge]):
    def __init__(self, data: dict[str, Attribute] | None = None):
        g: _Graph[str] = _Graph()
        if data is None:
            data = {}
        super().__init__(
            graph=g,
            node_fn=Node,
            edge_fn=Edge,
            data=data,
        )


def test_graph_is_serialized():
    g = Graph(data=dict(name="network", type="network"))
    g.add_node(node(name="node1", type="type1"))
    g.add_node(node(name="node2", type="type2"))
    g.add_edge(edge(src="node1", dst="node2"))
    assert g.data == {
        "name": "network",
        "type": "network",
        "nodes": {
            "node1": {"type": "type1"},
            "node2": {"type": "type2"},
        },
        "edges": {"node1": {"node2": {}}},
    }


def test_graph_has_required_fields():
    g = Graph(data=dict(name="network", type="network"))
    assert g.name == "network"


def test_can_retrieve_edges() -> None:
    g = (
        Graph()
        .add_node(name="x", data=dict(type="t"))
        .add_node(name="y", data=dict(type="t"))
        .add_edge(src="x", dst="y", data=dict())
    )
    edges = tuple(g.edges.values())
    assert edge(src="x", dst="y") == edges[0]


@pytest.fixture
def graph() -> Graph:
    g = (
        Graph()
        .add_node(name="x", data=dict(type="t"))
        .add_node(name="y", data=dict(type="t"))
        .add_node(name="z", data=dict(type="t"))
        .add_edge(src="x", dst="y", data={})
        .add_edge(src="y", dst="z", data=dict(extra="e"))
    )
    return g


def test_can_get_specific_edge(graph: Graph) -> None:
    e = graph.edges[("y", "z")]
    assert dict(extra="e") == e.attributes


def test_edges_is_read_only(graph: Graph) -> None:
    with pytest.raises(TypeError):
        graph.edges[("x", "z")] = edge("x", "z")  # type: ignore


def test_predecessors_of_z_is_y(graph: Graph) -> None:
    n = next(iter(graph.predecessors("z").values()))
    assert graph.nodes["y"] == n


def test_successor_of_x_is_y(graph: Graph) -> None:
    n = next(iter(graph.successors("x").values()))
    assert graph.nodes["y"] == n


def test_is_json_serializable(graph: Graph) -> None:
    result = json.dumps(graph.as_dict())
    assert result == (
        r'{"nodes": {"x": {"type": "t"}, "y": {"type": "t"}, "z": {"type": "t"}}, "edges": {"x": {"y": {}}, "y": {"z": {"extra": "e"}}}}'
    )


def test_change_node_attribute() -> None:
    impl = Implementation(graph=_Graph())
    impl.add_node(node(name="a", type="t"))
    impl.nodes["a"].data.update({"x": "y"})
    assert "y" == impl.nodes["a"].attributes["x"]
