from .edge import Edge
from .graph import Graph
from .node import Node


def test_has_input_node():
    g = Graph()
    assert g.get_node("input") == Node(name="input", type="input")


def test_can_test_for_node_existence():
    g = Graph()
    g.add_node(Node(name="a", type="b"))
    f = Graph()
    assert g.has_node("a") and not f.has_node("a")


def test_can_get_successors():
    g = Graph()
    g.add_node(Node(name="a", type="b")).add_edge(Edge(src="input", sink="a"))
    s = tuple(g.get_successors("input"))[0]
    assert Node(name="a", type="b") == s


def test_can_get_predecessors():
    g = Graph()
    g.add_node(Node(name="a", type="b")).add_edge(Edge(src="input", sink="a"))
    p = tuple(g.get_predecessors("a"))[0]
    assert Node(name="input", type="input") == p


def test_can_serialize_graph():
    g = Graph()
    g.add_edge(Edge(src="input", sink="a", attributes=dict(indices=(2, 3)))).add_node(
        Node(name="input", type="input", attributes=dict(input_shape=(2, 4, 8)))
    ).add_node(Node("a", "b"))
    g.get_node("input").attributes["output_shape"] = (1, 1, 1)

    serialized = g.as_dict()
    expected = {
        "nodes": [
            {
                "name": "input",
                "type": "input",
                "input_shape": (2, 4, 8),
                "output_shape": (1, 1, 1),
            },
            {"name": "output", "type": "output"},
            {"name": "a", "type": "b"},
        ],
        "edges": [{"src": "input", "sink": "a", "indices": (2, 3)}],
    }
    assert serialized == expected
