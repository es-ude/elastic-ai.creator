from .utils import build_graph_from_dict, find_matches

"""
Test List:

 * if adding a function to find all subgraphs, we need to check that
   matches already found are not discarded again. Because the implementation
   will try to visit matches that are already completed, which might cause
   them to be removed from the next list of potential matches.

"""


def test_find_identical_graph_with_single_node():
    g = build_graph_from_dict(
        {
            ("a", "a_t"): [],
        }
    )
    p = build_graph_from_dict(
        {
            ("a", "a_t"): [],
        }
    )
    assert find_matches(g, p) == [{"a": "a"}]


def test_find_identical_graph_with_multiple_nodes():
    g = build_graph_from_dict(
        {
            ("a", "a_t"): ["b"],
            ("b", "b_t"): [],
        }
    )
    p = build_graph_from_dict(
        {
            ("a", "a_t"): ["b"],
            ("b", "b_t"): [],
        }
    )
    assert find_matches(g, p) == [{"a": "a", "b": "b"}]


def test_find_graph_by_type():
    g = build_graph_from_dict(
        {
            ("a", "a_t"): ["b"],
            ("b", "b_t"): [],
        }
    )
    p = build_graph_from_dict(
        {
            ("0", "a_t"): ["1"],
            ("1", "b_t"): [],
        }
    )
    assert find_matches(g, p) == [{"0": "a", "1": "b"}]


def test_find_aba_patter_two_times():
    g = build_graph_from_dict(
        {
            ("a", "a_t"): ["b", "c"],
            ("b", "b_t"): ["d"],
            ("c", "c_t"): ["d"],
            ("d", "a_t"): ["e"],
            ("e", "b_t"): [],
        }
    )

    pattern = build_graph_from_dict(
        {
            ("0", "a_t"): ["1"],
            ("1", "b_t"): [],
        }
    )

    matches = find_matches(g, pattern)

    assert matches == [
        {"0": "d", "1": "e"},
        {"0": "a", "1": "b"},
    ]


def test_find_pattern_if_we_have_to_search_for_predecessors():
    g = build_graph_from_dict(
        {("a", "a_t"): ["c"], ("b", "b_t"): ["c"], ("c", "c_t"): []}
    )
    p = build_graph_from_dict(
        {("0", "a_t"): ["2"], ("1", "b_t"): ["2"], ("2", "c_t"): []}
    )
    assert find_matches(g, p) == [{"0": "a", "1": "b", "2": "c"}]
