import pytest

import elasticai.creator.graph as gr
from elasticai.creator.graph.vf2 import match


@pytest.fixture
def create_graph():
    def create():
        return gr.BaseGraph()

    return create


def add_path(graph: gr.Graph, path: str):
    node_path = path.split("->")
    for a, b in zip(node_path[:-1], node_path[1:]):
        graph.add_edge(a, b)


def test_match_single_edge(create_graph):
    p = create_graph()
    add_path(p, "a->b")
    g = create_graph()
    add_path(g, "0->1")
    assert match(p, g) == {"a": "0", "b": "1"}


def test_match_single_edge_two_times(create_graph):
    p = create_graph()
    add_path(p, "a->b")
    g = create_graph()
    add_path(g, "0->1->2")
    assert match(p, g) in [{"a": "0", "b": "1"}, {"a": "1", "b": "2"}]


def test_double_edge(create_graph):
    p = create_graph()
    add_path(p, "a->b->c")
    g = create_graph()
    add_path(g, "0->1->2")
    assert match(p, g) == {"a": "0", "b": "1", "c": "2"}


def test_double_out_edge(create_graph):
    p = create_graph()
    add_path(p, "a->b")
    g = create_graph()
    for path in ["0->1", "0->2"]:
        add_path(g, path)

    assert match(p, g) in [
        {"a": "0", "b": "1"},
        {"a": "0", "b": "2"},
    ]


def test_donot_match_partially_with_single_edge(create_graph):
    p = create_graph()
    add_path(p, "a->b->c")
    g = create_graph()
    add_path(g, "0->1")
    assert match(p, g) == dict()


def test_donot_match_partially_four_to_three_edges(create_graph):
    p = create_graph()
    add_path(p, "a->b->c->d->e")
    g = create_graph()
    add_path(g, "0->1->2->3")
    assert match(p, g) == dict()


def test_match_acyclic_pattern_to_cyclic_graph(create_graph):
    p = create_graph()
    add_path(p, "a->b")
    g = create_graph()
    add_path(g, "0->1->0")
    print(g.successors)
    print(p.successors)
    print(p.predecessors)
    assert match(p, g) in [{"a": "0", "b": "1"}, {"a": "1", "b": "0"}]


def test_match_cyclic_pattern_to_cyclic_graph(create_graph):
    p = create_graph()
    add_path(p, "a->b->a")
    g = create_graph()
    add_path(g, "0->1->0")
    assert match(p, g) in [{"a": "0", "b": "1"}, {"a": "1", "b": "0"}]


def test_node_constraint_can_prevent_first_match(create_graph):
    p = create_graph()
    add_path(p, "a->b")
    g = create_graph()
    add_path(g, "0->1->2")
    assert match(p, g, lambda pn, gn: gn in ("1", "2")) == {"a": "1", "b": "2"}


def test_match_diamond_pattern(create_graph):
    r"""
    a-b-c-e
     \   /
      d--

    1-2-3-4
     \   /
      6--
    """
    p = create_graph()
    add_path(p, "a->b->c->e")
    add_path(p, "a->d->e")
    g = create_graph()
    add_path(g, "1->2->3->4")
    add_path(g, "1->6->4")
    print(p.successors)
    print(g.successors)
    assert match(p, g) == {"a": "1", "b": "2", "c": "3", "d": "6", "e": "4"}
