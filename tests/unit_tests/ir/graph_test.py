import pytest

from elasticai.creator.ir import Edge, Node, edge, node
from elasticai.creator.ir import Implementation as _Graph


class Graph(_Graph[Node, Edge]):
    name: str
    type: str


def test_graph_is_serialized():
    g = Graph(data=dict(name="network", type="network"), node_fn=Node, edge_fn=Edge)
    g.add_node(node(name="node1", type="type1"))
    g.add_node(node(name="node2", type="type2"))
    g.add_edge(edge(src="node1", sink="node2"))
    assert g.data == {
        "name": "network",
        "type": "network",
        "nodes": {
            "node1": {"name": "node1", "type": "type1"},
            "node2": {"name": "node2", "type": "type2"},
        },
        "edges": {
            ("node1", "node2"): {"src": "node1", "sink": "node2"},
        },
    }


def test_graph_has_required_fields():
    g = Graph(data=dict(name="network", type="network"), node_fn=Node, edge_fn=Edge)
    assert g.name == "network"


def test_can_retrieve_edges() -> None:
    g = _Graph(
        nodes=(node(name="x", type="t"), node(name="y", type="t")),
        edges=(edge(src="x", sink="y"),),
    )
    edges = tuple(g.edges.values())
    assert edge(src="x", sink="y") == edges[0]


@pytest.fixture
def graph() -> _Graph:
    def _node(name: str):
        return node(name=name, type="t")

    g = _Graph(
        nodes=(_node("x"), _node("y"), _node("z")),
        edges=(edge("x", "y"), edge("y", "z", attributes=dict(extra="e"))),
    )
    return g


def test_can_get_specific_edge(graph: _Graph) -> None:
    e = graph.edges[("y", "z")]
    assert dict(extra="e") == e.attributes


def test_edges_is_read_only(graph: _Graph) -> None:
    with pytest.raises(TypeError):
        graph.edges[("x", "z")] = edge("x", "z")  # type: ignore


def test_predecessors_of_z_is_y(graph: _Graph) -> None:
    n = next(iter(graph.predecessors("z").values()))
    assert graph.nodes["y"] == n


def test_successor_of_x_is_y(graph: _Graph) -> None:
    n = next(iter(graph.successors("x").values()))
    assert graph.nodes["y"] == n
