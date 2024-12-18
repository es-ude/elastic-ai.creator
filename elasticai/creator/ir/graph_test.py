import pytest

from .core import edge, node
from .graph import Graph


def test_can_retrieve_edges() -> None:
    g = Graph(
        nodes=(node(name="x", type="t"), node(name="y", type="t")),
        edges=(edge(src="x", sink="y"),),
    )
    edges = tuple(g.edges.values())
    assert edge(src="x", sink="y") == edges[0]


@pytest.fixture
def graph() -> Graph:
    def _node(name: str):
        return node(name=name, type="t")

    g = Graph(
        nodes=(_node("x"), _node("y"), _node("z")),
        edges=(edge("x", "y"), edge("y", "z", attributes=dict(extra="e"))),
    )
    return g


def test_can_get_specific_edge(graph) -> None:
    e = graph.edges[("y", "z")]
    assert dict(extra="e") == e.attributes


def test_edges_is_read_only(graph) -> None:
    with pytest.raises(TypeError):
        graph.edges[("x", "z")] = edge("x", "z")


def test_predecessors_of_z_is_y(graph) -> None:
    n = next(iter(graph.predecessors("z").values()))
    assert graph.nodes["y"] is n


def test_successor_of_x_is_y(graph) -> None:
    n = next(iter(graph.successors("x").values()))
    assert graph.nodes["y"] is n
