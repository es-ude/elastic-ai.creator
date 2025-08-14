import logging

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

    assert raised and i == 1, (
        "Expected to raise DanglingEdgeError when rewriting second match, but did not."
    )


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


def create_real_world_graph_with_data() -> tuple[
    gr.Graph[str], dict[str, dict[str, str]]
]:
    data = {
        "input_1": {
            "type": "input",
        },
        "conv0_0": {
            "type": "conv1d",
        },
        "conv0_1": {
            "type": "maxpool1d",
        },
        "conv0_2": {
            "type": "batchnorm1d",
        },
        "conv0_3": {
            "type": "binarize",
        },
        "conv1_0": {
            "type": "conv1d",
        },
        "conv1_1": {
            "type": "maxpool1d",
        },
        "conv1_2": {
            "type": "batchnorm1d",
        },
        "conv1_3": {
            "type": "binarize",
        },
        "conv2_0": {
            "type": "conv1d",
        },
        "conv2_1": {
            "type": "maxpool1d",
        },
        "conv2_2": {
            "type": "batchnorm1d",
        },
        "conv2_3": {
            "type": "binarize",
        },
        "conv3_0": {
            "type": "conv1d",
        },
        "conv3_1": {
            "type": "maxpool1d",
        },
        "conv3_2": {
            "type": "batchnorm1d",
        },
        "conv3_3": {
            "type": "binarize",
        },
        "conv4_0": {
            "type": "conv1d",
        },
        "conv4_1": {
            "type": "maxpool1d",
        },
        "conv4_2": {
            "type": "batchnorm1d",
        },
        "conv4_3": {
            "type": "binarize",
        },
        "flatten": {
            "type": "flatten",
        },
        "lin": {
            "type": "linear",
        },
        "sigmoid": {
            "type": "sigmoid",
        },
        "flatten1": {
            "type": "flatten",
        },
        "output": {
            "type": "output",
        },
    }
    g: gr.Graph[str] = gr.BaseGraph()
    original_edges = {
        "input_1": {"conv0_0"},
        "conv0_0": {"conv0_1"},
        "conv0_1": {"conv0_2"},
        "conv0_2": {"conv0_3"},
        "conv0_3": {"conv1_0"},
        "conv1_0": {"conv1_1"},
        "conv1_1": {"conv1_2"},
        "conv1_2": {"conv1_3"},
        "conv1_3": {"conv2_0"},
        "conv2_0": {"conv2_1"},
        "conv2_1": {"conv2_2"},
        "conv2_2": {"conv2_3"},
        "conv2_3": {"conv3_0"},
        "conv3_0": {"conv3_1"},
        "conv3_1": {"conv3_2"},
        "conv3_2": {"conv3_3"},
        "conv3_3": {"conv4_0"},
        "conv4_0": {"conv4_1"},
        "conv4_1": {"conv4_2"},
        "conv4_2": {"conv4_3"},
        "conv4_3": {"flatten"},
        "flatten": {"lin"},
        "lin": {"sigmoid"},
        "sigmoid": {"flatten1"},
        "flatten1": {"output"},
    }
    for src in original_edges:
        for dst in original_edges[src]:
            g.add_edge(src, dst)

    return g, data


@pytest.fixture
def expected_edges_after_complex_reordering():
    expected_edges = {
        "input_1": {"conv0_0"},
        "conv0_0": {"batchnorm1d"},
        "batchnorm1d": {"binarize"},
        "binarize": {"maxpool1d"},
        "maxpool1d": {"conv1_0"},
        "conv1_0": {"batchnorm1d_1"},
        "batchnorm1d_1": {"binarize_1"},
        "binarize_1": {"maxpool1d_1"},
        "maxpool1d_1": {"conv2_0"},
        "conv2_0": {"batchnorm1d_2"},
        "batchnorm1d_2": {"binarize_2"},
        "binarize_2": {"maxpool1d_2"},
        "maxpool1d_2": {"conv3_0"},
        "conv3_0": {"batchnorm1d_3"},
        "batchnorm1d_3": {"binarize_3"},
        "binarize_3": {"maxpool1d_3"},
        "maxpool1d_3": {"conv4_0"},
        "conv4_0": {"batchnorm1d_4"},
        "batchnorm1d_4": {"binarize_4"},
        "binarize_4": {"maxpool1d_4"},
        "maxpool1d_4": {"flatten"},
        "flatten": {"lin"},
        "lin": {"sigmoid"},
        "sigmoid": {"flatten1"},
        "flatten1": {"output"},
        "output": set(),
    }
    return expected_edges


def test_can_reorder_complex_layer_sequence(expected_edges_after_complex_reordering):
    logger = logging.getLogger("elasticai.creator.graph")

    def make_pattern(data):
        """Also consider the corresponding constraint for this pattern.

        The pattern is defined in such a way that we declare potentially
        matched convolutional layers to be part of the interface:
        'start', 'end'. Thus, we "promise" that we will not edit
        convolutional layers. Instead, we will only reorder
        maxpool1d, batchnorm1d, and binarize layers.
        """
        pattern = gr.BaseGraph()
        pattern_sequence = [
            "start",
            "maxpool1d",
            "batchnorm1d",
            "binarize",
            "end",
        ]

        for src, dst in zip(pattern_sequence[:-1], pattern_sequence[1:]):
            pattern.add_edge(src, dst)

        def constraint(pattern_node: str, graph_node: str) -> bool:
            match pattern_node:
                case "start":
                    return data[graph_node]["type"] in ("input", "conv1d")
                case "end":
                    return data[graph_node]["type"] in ("output", "conv1d", "flatten")
                case _:
                    return data[graph_node]["type"] == pattern_node

        return pattern, constraint

    def make_replacement():
        replacement_sequence = [
            "start",
            "batchnorm1d",
            "binarize",
            "maxpool1d",
            "end",
        ]
        replacement = gr.BaseGraph()

        for src, dst in zip(replacement_sequence[:-1], replacement_sequence[1:]):
            replacement.add_edge(src, dst)
        return replacement

    g, data = create_real_world_graph_with_data()
    pattern, constraint = make_pattern(data)
    matches = list(
        gr.find_all_subgraphs(
            pattern=pattern,
            graph=g,
            node_constraint=constraint,
        )
    )

    def log(msg):
        logger.log(logging.DEBUG, msg)

    log(f"Found {len(matches)} matches")
    for match in matches:
        log(f"Match: {match}")

    lhs = {"i0": "start", "i1": "end"}
    safe_matches = list(
        gr.get_rewriteable_matches(g, matches, interface_nodes=lhs.values())
    )

    log(f"Safe matches: {len(safe_matches)}")
    for match in safe_matches:
        log(f"Match: {match}")

    def rewrite_all_matches(g, matches, replacement, lhs, rhs):
        result = g
        for match in matches:
            result, replacement_map = gr.rewrite(
                replacement=replacement, original=result, match=match, lhs=lhs, rhs=rhs
            )
        return result

    def visit_nodes(graph, node, visited):
        if node not in visited:
            visited.add(node)
            log(node)
            for successor in graph.successors[node]:
                visit_nodes(graph, successor, visited)

    replacement = make_replacement()
    rhs = lhs
    result = rewrite_all_matches(g, safe_matches, replacement, lhs, rhs)

    visit_nodes(result, "input_1", set())
    assert expected_edges_after_complex_reordering == dict(result.successors)
