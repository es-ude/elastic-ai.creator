from elasticai.creator import graph as gr
from elasticai.creator.graph import produces_dangling_edge


def build_graph(adjacency_list):
    g = gr.BaseGraph()
    for src in adjacency_list:
        for dst in adjacency_list[src]:
            g.add_edge(src, dst)
    return g


def test_catch_dangling_edge_d_b():
    g = build_graph(
        {
            "a": ["b"],
            "b": ["c"],
            "d": ["b"],
        }
    )
    match = {"0": "a", "1": "b", "2": "c"}
    lhs = {"i0": "0", "i1": "2"}
    assert produces_dangling_edge(g, match, lhs.values())


def test_catch_dangling_edge_b_d():
    g = build_graph(
        {
            "a": ["b"],
            "b": ["c", "d"],
        }
    )
    match = {"0": "a", "1": "b", "2": "c"}
    lhs = {"i0": "0", "i1": "2"}
    assert produces_dangling_edge(g, match, lhs.values())


def test_let_pass_if_successor_b_is_in_interface():
    g = build_graph(
        {
            "a": ["b"],
            "b": ["c"],
            "d": ["b"],
        }
    )
    match = {"0": "a", "1": "b", "2": "c"}
    lhs = {"i0": "0", "i1": "1"}
    assert not produces_dangling_edge(g, match, lhs.values())


def test_let_pass_if_predecessor_b_is_in_interface():
    g = build_graph(
        {
            "a": ["b"],
            "b": ["c", "d"],
        }
    )
    match = {"0": "a", "1": "b", "2": "c"}
    lhs = {"i0": "0", "i1": "1"}
    assert not produces_dangling_edge(g, match, lhs.values())


def test_remove_second_match_if_it_overlaps_with_first():
    g = build_graph(
        {
            "a": ["b"],
            "b": ["c"],
            "c": ["d"],
        }
    )
    first_match = {"0": "a", "1": "b", "2": "c"}
    second_match = {"0": "b", "1": "c", "2": "d"}
    lhs = {"i0": "0", "i1": "2"}
    assert [first_match] == list(
        gr.get_rewriteable_matches(
            g,
            [first_match, second_match],
            lhs.values(),
        )
    )


def test_do_not_remove_second_match_if_overlap_only_occurs_in_interface():
    g = build_graph(
        {
            "a": ["b"],
            "b": ["c"],
            "c": ["d"],
            "d": ["e"],
        }
    )
    first_match = {"0": "a", "1": "b", "2": "c"}
    second_match = {"0": "c", "1": "d", "2": "e"}
    lhs = {"i0": "0", "i1": "2"}
    assert [first_match, second_match] == list(
        gr.get_rewriteable_matches(
            g,
            [first_match, second_match],
            lhs.values(),
        )
    )


def test_pattern_seq_with_2_interface_nodes():
    g = build_graph(
        {
            "a": ["b"],
            "b": ["c"],
            "c": ["d"],
            "d": ["e"],
            "e": ["f"],
            "f": ["g"],
            "g": ["h"],
        }
    )
    p = build_graph(
        {
            "0": ["1"],
            "1": ["2"],
            "2": ["3"],
            "3": ["4"],
        }
    )
    matches = gr.find_all_subgraphs(
        pattern=p, graph=g, node_constraint=lambda _, __: True
    )
    lhs = {"i0": "0", "i1": "1", "i2": "3", "i3": "4"}
    expected_leftover_matches = []
    for sequence in [("a", "b", "c", "d", "e"), ("d", "e", "f", "g", "h")]:
        expected_leftover_matches.append(
            {f"{i}": node for i, node in enumerate(sequence)}
        )
    assert expected_leftover_matches == list(
        gr.get_rewriteable_matches(
            g,
            matches,
            lhs.values(),
        )
    )
