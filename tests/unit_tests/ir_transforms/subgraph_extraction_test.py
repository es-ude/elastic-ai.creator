import pytest

from elasticai.creator import graph, ir
from elasticai.creator import ir_transforms as itr


class TestExtractTwoNodesFromImpl:
    @pytest.fixture
    def impl(self):
        return (
            ir.Implementation(
                data=dict(name="impl"),
                graph=graph.BaseGraph(),
            )
            .add_node(name="input", type="input")
            .add_node(name="output", type="output")
            .add_node(name="a", type="t")
            .add_node(name="b", type="t")
            .add_edge(src="input", dst="a")
            .add_edge(src="a", dst="b")
            .add_edge(src="b", dst="output")
        )

    @pytest.fixture
    def pattern(self):
        return (
            ir.Implementation(
                data=dict(name="pattern", type="p"),
                graph=graph.BaseGraph(),
            )
            .add_node(name="input", type="any")
            .add_node(name="output", type="any")
            .add_node(name="a", type="t")
            .add_node(name="b", type="t")
            .add_edge(src="input", dst="a")
            .add_edge(src="a", dst="b")
            .add_edge(src="b", dst="output")
        )

    @pytest.fixture
    def constraint(self):
        def _constraint(pattern_node: ir.Node, graph_node: ir.Node) -> bool:
            return pattern_node.type == "any" or pattern_node.type == graph_node.type

        return _constraint

    @pytest.fixture
    def extraction_result(
        self,
        impl: ir.Implementation[ir.Node, ir.Edge],
        pattern: ir.Implementation[ir.Node, ir.Edge],
        constraint,
    ):
        return itr.SubgraphExtractor(pattern, node_constraint=constraint).extract(impl)

    @pytest.fixture
    def new_graph(
        self,
        extraction_result: tuple[
            ir.Implementation[ir.Node, ir.Edge], ir.Implementation[ir.Node, ir.Edge]
        ],
    ):
        return extraction_result[0]

    @pytest.fixture
    def subgraph(
        self,
        extraction_result: tuple[
            ir.Implementation[ir.Node, ir.Edge], ir.Implementation[ir.Node, ir.Edge]
        ],
    ):
        return extraction_result[1]

    @pytest.fixture
    def test_new_graph_has_all_required_nodes(
        self, new_graph: ir.Implementation[ir.Node, ir.Edge]
    ):
        assert new_graph.data["nodes"] == {
            "input": {"type": "input", "name": "input"},
            "output": {"type": "output", "name": "output"},
            "pattern": {"name": "pattern", "type": "p", "implementation": "pattern"},
        }

    def test_new_graph_has_all_required_edges(
        self,
        new_graph: ir.Implementation[ir.Node, ir.Edge],
    ):
        assert set(new_graph.edges.keys()) == {
            ("input", "pattern"),
            ("pattern", "output"),
        }

    def test_subgraph_has_all_required_nodes(
        self,
        subgraph: ir.Implementation[ir.Node, ir.Edge],
    ):
        assert subgraph.data["nodes"] == {
            "input": {"name": "input", "type": "input"},
            "output": {"name": "output", "type": "output"},
            "a": {"name": "a", "type": "t"},
            "b": {"name": "b", "type": "t"},
        }

    def test_subgraph_has_all_required_edges(
        self,
        subgraph: ir.Implementation[ir.Node, ir.Edge],
    ):
        assert set(subgraph.edges.keys()) == {
            ("input", "a"),
            ("a", "b"),
            ("b", "output"),
        }
