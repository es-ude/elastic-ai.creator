import pytest

from .utils import build_graph_from_dict, get_rewriter


def test_can_replace_single_node():
    graph = build_graph_from_dict(
        {("a", "a_t"): ["b"], ("b", "b_t"): ["c"], ("c", "c_t"): []}
    )
    pattern = build_graph_from_dict(
        {("0", "a_t"): ["1"], ("1", "b_t"): ["2"], ("2", "c_t"): []}
    )
    interface = build_graph_from_dict({("i0", ""): [], ("i1", ""): []})
    replacement = build_graph_from_dict(
        {("r0", "a_t"): ["r1"], ("r1", "r_t"): ["r2"], ("r2", "c_t"): []}
    )

    def lhs(node: str) -> str:
        return {"i0": "0", "i1": "2"}[node]

    def rhs(node: str) -> str:
        return {"i0": "r0", "i1": "r2"}[node]

    rewrite = get_rewriter(pattern, interface, replacement, lhs, rhs)

    expected = build_graph_from_dict(
        {
            ("c", "c_t"): [],
            ("a", "a_t"): ["r1"],
            ("r1", "r_t"): ["c"],
        }
    )
    assert rewrite(graph).as_dict() == expected.as_dict()


def test_create_new_name_for_replacement_in_case_of_conflict():
    graph = build_graph_from_dict(
        {("a", "a_t"): ["b"], ("b", "b_t"): ["c"], ("c", "c_t"): []}
    )
    pattern = build_graph_from_dict(
        {("0", "a_t"): ["1"], ("1", "b_t"): ["2"], ("2", "c_t"): []}
    )
    interface = build_graph_from_dict({("i0", ""): [], ("i1", ""): []})
    replacement = build_graph_from_dict(
        {("r0", "a_t"): ["a"], ("a", "r_t"): ["r2"], ("r2", "c_t"): []}
    )

    def lhs(node: str) -> str:
        return {"i0": "0", "i1": "2"}[node]

    def rhs(node: str) -> str:
        return {"i0": "r0", "i1": "r2"}[node]

    rewrite = get_rewriter(pattern, interface, replacement, lhs, rhs)

    expected = build_graph_from_dict(
        {
            ("c", "c_t"): [],
            ("a", "a_t"): ["a_1"],
            ("a_1", "r_t"): ["c"],
        }
    )
    assert rewrite(graph).as_dict() == expected.as_dict()


def test_raise_error_if_rhs_is_not_injective():
    pattern = build_graph_from_dict(
        {("0", "a_t"): ["1"], ("1", "b_t"): ["2"], ("2", "c_t"): []}
    )
    interface = build_graph_from_dict({("i0", ""): [], ("i1", ""): []})
    replacement = build_graph_from_dict(
        {("r0", "a_t"): ["a"], ("a", "r_t"): ["r2"], ("r2", "c_t"): []}
    )

    def lhs(node: str) -> str:
        return {"i0": "0", "i1": "2"}[node]

    def rhs(node: str) -> str:
        return {"i0": "r0", "i1": "r0"}[node]

    pytest.raises(
        ValueError,
        get_rewriter,
        pattern,
        interface,
        replacement,
        lhs,
        rhs,
    )


def test_can_replace_one_of_two_matches():
    graph = build_graph_from_dict(
        {
            ("a", "a_t"): ["b"],
            ("b", "b_t"): ["c"],
            ("c", "a_t"): ["d"],
            ("d", "b_t"): ["e"],
            ("e", "a_t"): [],
        }
    )
    pattern = build_graph_from_dict(
        {("0", "a_t"): ["1"], ("1", "b_t"): ["2"], ("2", "a_t"): []}
    )
    interface = build_graph_from_dict({("i0", ""): [], ("i1", ""): []})
    replacement = build_graph_from_dict(
        {("r0", "a_t"): ["r1"], ("r1", "r_t"): ["r2"], ("r2", "c_t"): []}
    )

    def lhs(node: str) -> str:
        return {"i0": "0", "i1": "2"}[node]

    def rhs(node: str) -> str:
        return {"i0": "r0", "i1": "r2"}[node]

    rewrite = get_rewriter(pattern, interface, replacement, lhs, rhs)

    expected = build_graph_from_dict(
        {
            ("c", "a_t"): ["r1"],
            ("a", "a_t"): ["b"],
            ("b", "b_t"): ["c"],
            ("r1", "r_t"): ["e"],
            ("e", "a_t"): [],
        }
    )
    actual = rewrite(graph)
    assert set(actual.edges) == set(expected.edges)
    assert set(actual.nodes) == set(expected.nodes)
