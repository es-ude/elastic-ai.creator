from collections.abc import Callable

import pytest

from elasticai.creator.graph.subgraph_matching import find_all_matches
from elasticai.creator.ir.graph import AdjacencyMap, Graph, GraphImpl


@pytest.fixture
def new_graph() -> Callable[[], Graph[str, str]]:
    def new():
        return GraphImpl(lambda: "default")

    return new


def adding_node_does_not_modify_original_graph(new_graph) -> None:
    g1: Graph[str, str] = new_graph
    g2 = g1.add_node("a")

    assert "a" not in g1.successors
    assert "a" in g2.successors


def test_nested_mapping_can_merge_with_other_nested_mapping() -> None:
    a = AdjacencyMap({"x": {"y": 1}})
    b = AdjacencyMap({"x": {"z": 2}})
    c = a.join(b)
    assert c["x"] == {"y": 1, "z": 2}


def test_updating_nested_mapping_with_empty_submaps_adds_empty_submaps() -> None:
    nm1 = AdjacencyMap({"a": {"x": 1}})
    nm2 = nm1.join({"b": {}})

    assert "b" in nm2
    assert nm2["b"] == {}


def test_can_drop_key_pair_from_nested_mapping() -> None:
    nm1 = AdjacencyMap({"a": {"x": 1, "y": 2}, "b": {"z": 3}})
    nm2 = nm1.drop("a", "y")
    nm3 = nm2.drop("a", "x")

    assert nm2["a"] == {"x": 1}
    assert nm3["a"] == dict()


def test_can_drop_key_from_nested_mapping() -> None:
    nm1 = AdjacencyMap({"a": {"x": 1, "y": 2}, "x": {"z": 3}})
    nm2 = nm1.drop("x")
    assert "x" not in nm2["a"]
    assert "x" not in nm2


def test_can_create_new_nested_mapping_by_merging_with_dict() -> None:
    a: AdjacencyMap[str, int] = AdjacencyMap({})
    b = a.join({"x": {"y": 1}})
    assert b == {"x": {"y": 1}}


def test_can_add_edges_without_attributes_to_graph(new_graph) -> None:
    g1: Graph[str, str] = new_graph()
    g2 = g1.add_edges(("a", "b"))

    assert "default" == g2.successors["a"]["b"]


def adding_edge_does_not_modify_original_graph(new_graph) -> None:
    g1: Graph[str, str] = new_graph()
    g2 = g1.add_edge("a", "b", "E")

    def iter_edges[N, E](graph):
        for src in graph.successors:
            for dst, attributes in graph.successors[src].items():
                yield src, dst, attributes

    assert ("a", "b") not in list(iter_edges(g1))
    assert ("a", "b") in list(iter_edges(g2))


def can_add_multiple_edges_at_once(new_graph) -> None:
    g1: Graph[str, str] = new_graph()
    g2 = g1.add_edges(
        ("a", "b", "A"),
        ("b", "c", "B"),
    )

    g2_edges = set()
    for src in g2.successors:
        for dst in g2.successors[src]:
            g2_edges.add((src, dst))

    assert ("a", "b") in g2_edges
    assert ("b", "c") in g2_edges


def test_adding_edge_adds_nodes_to_predecessors_and_successors(new_graph) -> None:
    g: Graph[str, str] = new_graph()
    g = g.add_edge("a", "b", "E")

    assert "a" in g.successors
    assert "a" in g.predecessors


def test_can_find_subgraphs(new_graph) -> None:
    g = new_graph().add_edges(("a", "b"), ("b", "c"), ("b", "d"))
    pattern = new_graph().add_edges(("a", "b"), ("b", "c"))

    def dict_list_to_set(
        ds,
    ):  # convert to sets because order of returned matches is not stable
        return set(tuple((k, v) for k, v in d.items()) for d in ds)

    matches = dict_list_to_set(find_all_matches(pattern, g))
    assert (
        dict_list_to_set(
            [{"a": "a", "b": "b", "c": "c"}, {"a": "a", "b": "b", "c": "d"}]
        )
        == matches
    )


def test_can_remove_edge(new_graph) -> None:
    g1 = new_graph().add_edges(("a", "b"), ("b", "c"))
    g2 = g1.remove_edge("a", "b")
    edges = set()
    for node in g2.successors:
        for succ in g2.successors[node]:
            edges.add((node, succ))
    for node in g2.predecessors:
        for pred in g2.predecessors[node]:
            edges.add((pred, node))
    assert edges == {("b", "c")}
