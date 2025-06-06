import pytest

import elasticai.creator.graph as gr
from elasticai.creator.graph.graph_rewriting import RewriteResult
from tests.unit_tests.graph.utils import (
    Graph,
    get_rewriter,
    get_rewriter_returning_full_result,
)

from .utils import (
    build_graph_from_dict,
)

"""Tests
- fail if removing pattern match leaves dangling edges
"""


def test_can_replace_single_node():
    graph = build_graph_from_dict(
        {("a", "a_t"): ["b"], ("b", "b_t"): ["c"], ("c", "c_t"): []}
    )
    pattern = build_graph_from_dict(
        {("0", "a_t"): ["1"], ("1", "b_t"): ["2"], ("2", "c_t"): []}
    )

    replacement = build_graph_from_dict(
        {("r0", "a_t"): ["r1"], ("r1", "r_t"): ["r2"], ("r2", "c_t"): []}
    )

    lhs = {"i0": "0", "i1": "2"}

    rhs = {"i0": "r0", "i1": "r2"}

    rewrite = get_rewriter(pattern, replacement, lhs, rhs)

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

    replacement = build_graph_from_dict(
        {("r0", "a_t"): ["a"], ("a", "r_t"): ["r2"], ("r2", "c_t"): []}
    )

    lhs = {"i0": "0", "i1": "2"}

    rhs = {"i0": "r0", "i1": "r2"}

    rewrite = get_rewriter(pattern, replacement, lhs, rhs)

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

    replacement = build_graph_from_dict(
        {("r0", "a_t"): ["a"], ("a", "r_t"): ["r2"], ("r2", "c_t"): []}
    )

    lhs = {"i0": "0", "i1": "2"}
    rhs = {"i0": "r0", "i1": "r0"}
    rewriter = get_rewriter(pattern, replacement, lhs, rhs)
    pytest.raises(
        ValueError,
        rewriter,
        build_graph_from_dict(
            {("a", "a_t"): ["b"], ("b", "b_t"): ["c"], ("c", "c_t"): []}
        ),
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

    replacement = build_graph_from_dict(
        {("r0", "a_t"): ["r1"], ("r1", "r_t"): ["r2"], ("r2", "c_t"): []}
    )

    lhs = {"i0": "0", "i1": "2"}
    rhs = {"i0": "r0", "i1": "r2"}

    rewrite = get_rewriter(pattern, replacement, lhs, rhs)

    expected = build_graph_from_dict(
        {
            ("c", "a_t"): ["d"],
            ("a", "a_t"): ["r1"],
            ("r1", "r_t"): ["c"],
            ("d", "b_t"): ["e"],
            ("e", "a_t"): [],
        }
    )
    actual = rewrite(graph)
    assert set(actual.edges) == set(expected.edges)
    assert set(actual.nodes) == set(expected.nodes)


def test_raise_error_if_rewrite_would_produce_dangling_edge():
    g = (
        gr.BaseGraph()
        .add_edge("a", "b")
        .add_edge("b", "c")
        .add_edge("c", "d")
        .add_edge("would_be_disconnected", "b")
    )
    raised = False
    try:
        gr.rewrite(
            replacement=gr.BaseGraph().add_edge("i0", "r").add_edge("r", "i1"),
            original=g,
            match={"i0": "a", "p0": "b", "p1": "c", "i1": "d"},
            lhs={"i0": "i0", "i1": "i1"},
            rhs={"i0": "i0", "i1": "i1"},
        )
    except gr.DanglingEdgeError:
        raised = True

    assert raised


@pytest.fixture
def graph_for_multiple_matches():
    graph = gr.BaseGraph()
    sequence = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for e in zip(sequence[:-1], sequence[1:]):
        graph.add_edge(e[0], e[1])
    return graph


def test_replacing_all_matches_fails(graph_for_multiple_matches):
    sequence = ("a", "b", "c", "d", "e", "f", "g", "h")
    all_matches = list(zip(sequence[:-2], sequence[1:-1], sequence[2:]))
    all_matches = [{"0": x, "1": y, "2": z} for x, y, z in all_matches]
    replacement = gr.BaseGraph().add_edge("r0", "r1").add_edge("r1", "r2")
    lhs = {"i0": "0", "i1": "2"}
    rhs = {"i0": "r0", "i1": "r2"}
    result = graph_for_multiple_matches
    raised = False
    try:
        for i, match in enumerate(all_matches):
            result, replacement_map = gr.rewrite(
                replacement=replacement,
                original=result,
                match=match,
                lhs=lhs,
                rhs=rhs,
            )
    except gr.DanglingEdgeError:
        raised = True

    assert (
        raised and i == 1
    ), "Expected to raise DanglingEdgeError when rewriting second match, but did not."


def test_can_replace_multiple_matches(graph_for_multiple_matches):
    graph = graph_for_multiple_matches
    safe_matches = [("a", "b", "c"), ("c", "d", "e"), ("e", "f", "g")]
    safe_matches = [{"0": x, "1": y, "2": z} for x, y, z in safe_matches]
    replacement = gr.BaseGraph().add_edge("r0", "r1").add_edge("r1", "r2")
    result = graph
    for match in safe_matches:
        result, replacement_map = gr.rewrite(
            replacement=replacement,
            original=result,
            match=match,
            lhs={"i0": "0", "i1": "2"},
            rhs={"i0": "r0", "i1": "r2"},
        )
    expected_sequence = ["a", "r1", "c", "r1_1", "e", "r1_2", "g", "h"]
    expected_edges = set(zip(expected_sequence[:-1], expected_sequence[1:]))
    assert set(result.iter_edges()) == expected_edges


class TestRewriteResult:
    @pytest.fixture
    def graph(self):
        return build_graph_from_dict(
            {("a", "a_t"): ["b"], ("b", "b_t"): ["c"], ("c", "c_t"): []}
        )

    @pytest.fixture
    def pattern(self):
        return build_graph_from_dict(
            {("0", "a_t"): ["1"], ("1", "b_t"): ["2"], ("2", "c_t"): []}
        )

    @pytest.fixture
    def replacement(self):
        return build_graph_from_dict(
            {("r0", "a_t"): ["r1"], ("r1", "r_t"): ["r2"], ("r2", "c_t"): []}
        )

    @pytest.fixture
    def rhs_dict(self):
        return {"i0": "r0", "i1": "r2"}

    @pytest.fixture
    def result(
        self,
        graph: Graph,
        pattern: Graph,
        replacement: Graph,
        rhs_dict: dict[str, str],
    ):
        lhs = {"i0": "0", "i1": "2"}

        rhs = rhs_dict

        rewrite = get_rewriter_returning_full_result(pattern, replacement, lhs, rhs)
        result = rewrite(graph)
        return result

    def test_can_map_pattern_back_to_original_graph(self, result: RewriteResult):
        expected_pattern_to_original = {"0": "a", "1": "b", "2": "c"}

        assert result.pattern_to_original == expected_pattern_to_original

    def test_can_map_replacement_to_new_graph(self, result: RewriteResult):
        expected_replacement_to_new = {"r0": "a", "r1": "r1", "r2": "c"}

        assert result.replacement_to_new == expected_replacement_to_new

    def test_can_combine_attribute_from_replacement_and_original_graph(
        self,
        result: RewriteResult,
        graph: Graph,
        pattern: Graph,
        replacement: Graph,
        rhs_dict: dict[str, str],
    ):
        graph.data = {"a": "1", "b": "2", "c": "3"}
        replacement.data = {"r0": "100", "r1": "200", "r2": "300"}

        new_graph = Graph(result.new_graph, {})
        for repl_node in replacement.data:
            new_node = result.replacement_to_new[repl_node]
            original_node_with_interesting_value = result.pattern_to_original["1"]
            interesting_value = graph.data[original_node_with_interesting_value]
            replacement_value = replacement.data[repl_node]
            not_in_interface = repl_node not in rhs_dict.values()
            if not_in_interface:
                new_graph.data[new_node] = str(
                    int(replacement_value) + int(interesting_value)
                )
            else:
                new_graph.data[new_node] = graph.data[new_node]

        assert new_graph.data == {"a": "1", "r1": "202", "c": "3"}
