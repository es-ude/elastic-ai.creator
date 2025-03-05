from elasticai.creator import graph as gr
from elasticai.creator.graph import produces_dangling_edge


def test_catch_dangling_edge_d_b():
    g = gr.BaseGraph().add_edge("a", "b").add_edge("b", "c").add_edge("d", "b")
    match = {"0": "a", "1": "b", "2": "c"}
    lhs = {"i0": "0", "i1": "2"}
    assert produces_dangling_edge(g, match, lhs.values())


def test_catch_dangling_edge_b_d():
    g = gr.BaseGraph().add_edge("a", "b").add_edge("b", "c").add_edge("b", "d")
    match = {"0": "a", "1": "b", "2": "c"}
    lhs = {"i0": "0", "i1": "2"}
    assert produces_dangling_edge(g, match, lhs.values())


def test_let_pass_if_successor_b_is_in_interface():
    g = gr.BaseGraph().add_edge("a", "b").add_edge("b", "c").add_edge("d", "b")
    match = {"0": "a", "1": "b", "2": "c"}
    lhs = {"i0": "0", "i1": "1"}
    assert not produces_dangling_edge(g, match, lhs.values())


def test_let_pass_if_predecessor_b_is_in_interface():
    g = gr.BaseGraph().add_edge("a", "b").add_edge("b", "c").add_edge("b", "d")
    match = {"0": "a", "1": "b", "2": "c"}
    lhs = {"i0": "0", "i1": "1"}
    assert not produces_dangling_edge(g, match, lhs.values())
