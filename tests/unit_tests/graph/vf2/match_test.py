from collections.abc import Callable

import pytest

import elasticai.creator.graph as gr
from elasticai.creator.graph.base_graph import BaseGraph
from elasticai.creator.graph.graph import Graph
from elasticai.creator.graph.vf2 import find_all_matches, match
from elasticai.creator.graph.vf2.matching import MatchError


@pytest.fixture
def create_graph():
    def create():
        return gr.BaseGraph()

    return create


def add_path(graph: gr.Graph, path: str):
    node_path = path.split("->")
    for a, b in zip(node_path[:-1], node_path[1:]):
        graph.add_edge(a, b)


def test_match_single_edge(create_graph: Callable[[], BaseGraph]):
    p = create_graph()
    add_path(p, "a->b")
    g = create_graph()
    add_path(g, "0->1")
    assert match(p, g) == {"a": "0", "b": "1"}


def test_match_single_edge_two_times(create_graph: Callable[[], BaseGraph]):
    p = create_graph()
    add_path(p, "a->b")
    g = create_graph()
    add_path(g, "0->1->2->3")
    assert find_all_matches(p, g) == [
        {"a": "0", "b": "1"},
        {"a": "1", "b": "2"},
        {"a": "2", "b": "3"},
    ]


def test_graph_matches_itself(create_graph: Callable[[], Graph]):
    g = create_graph()
    add_path(g, "1->2")
    add_path(g, "1->3")
    print(g.successors)

    match(g, g)


def test_double_edge(create_graph: Callable[[], BaseGraph]):
    p = create_graph()
    add_path(p, "a->b->c")
    g = create_graph()
    add_path(g, "0->1->2")
    assert match(p, g) == {"a": "0", "b": "1", "c": "2"}


def test_double_out_edge(create_graph: Callable[[], BaseGraph]):
    p = create_graph()
    add_path(p, "a->b")
    g = create_graph()
    for path in ["0->1", "0->2"]:
        add_path(g, path)
    assert match(p, g) in [
        {"a": "0", "b": "1"},
        {"a": "0", "b": "2"},
    ]


def test_donot_match_partially_with_single_edge(create_graph: Callable[[], BaseGraph]):
    p = create_graph()
    add_path(p, "a->b->c")
    g = create_graph()
    add_path(g, "0->1")
    pytest.raises(MatchError, match, p, g)


def test_donot_match_partially_four_to_three_edges(
    create_graph: Callable[[], BaseGraph],
):
    p = create_graph()
    add_path(p, "a->b->c->d->e")
    g = create_graph()
    add_path(g, "0->1->2->3")
    pytest.raises(MatchError, match, p, g)


def test_match_acyclic_pattern_to_cyclic_graph(create_graph: Callable[[], BaseGraph]):
    p = create_graph()
    add_path(p, "a->b")
    g = create_graph()
    add_path(g, "0->1->0")

    assert find_all_matches(p, g) == []


def test_match_cyclic_pattern_to_cyclic_graph(create_graph: Callable[[], BaseGraph]):
    p = create_graph()
    add_path(p, "a->b->a")
    g = create_graph()
    add_path(g, "0->1->0")
    assert match(p, g) in [{"a": "0", "b": "1"}, {"a": "1", "b": "0"}]


def test_node_constraint_can_prevent_first_match(create_graph: Callable[[], BaseGraph]):
    p = create_graph()
    add_path(p, "a->b")
    g = create_graph()
    add_path(g, "0->1->2")
    assert match(p, g, lambda pn, gn: gn in ("1", "2")) == {"a": "1", "b": "2"}


def test_match_diamond_pattern(create_graph: Callable[[], BaseGraph]):
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
    assert match(p, g) == {"a": "1", "b": "2", "c": "3", "d": "6", "e": "4"}
