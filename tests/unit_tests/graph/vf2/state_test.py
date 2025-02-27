from elasticai.creator import graph as gr
from elasticai.creator.graph.vf2.state import State


def test_update_in_nodes():
    state = State(graph=gr.BaseGraph().add_edge("a", "b"))
    state.add_pair("a", "first")
    state.next_depth()
    in_nodes = [state.order_back[i] for i, d in enumerate(state.in_nodes) if d == 1]
    out_nodes = [state.order_back[i] for i, d in enumerate(state.out_nodes) if d == 1]
    assert (in_nodes, out_nodes) == ([], ["b"])


def test_update_out_nodes():
    state = State(graph=gr.BaseGraph().add_edge("a", "b"))
    state.add_pair("b", "second")
    state.next_depth()
    in_nodes = [state.order_back[i] for i, d in enumerate(state.in_nodes) if d == 1]
    out_nodes = [state.order_back[i] for i, d in enumerate(state.out_nodes) if d == 1]
    assert (in_nodes, out_nodes) == (["a"], [])


def test_get_partial_succ():
    s = State(graph=gr.BaseGraph().add_edge("a", "b"))
    s.add_pair("b", "second")
    s.next_depth()
    assert s.partial_successors("a") == {"b"}


def test_get_partial_pred():
    s = State(gr.BaseGraph().add_edge("a", "b"))
    s.add_pair("a", "first")
    s.next_depth()
    assert s.partial_predecessors("b") == {"a"}


def test_get_unseen_nodes():
    s = State(gr.BaseGraph().add_edge("a", "b").add_edge("b", "c"))
    s.add_pair("a", "first")
    s.next_depth()
    assert s.unseen_nodes() == {"c"}


def test_unseen_successors():
    s = State(
        gr.BaseGraph()
        .add_edge("a", "b")
        .add_edge("b", "c")
        .add_edge("b", "d")
        .add_edge("a", "d")
    )
    s.add_pair("a", "first")
    s.next_depth()
    assert s.unseen_successors("b") == {"c"}
