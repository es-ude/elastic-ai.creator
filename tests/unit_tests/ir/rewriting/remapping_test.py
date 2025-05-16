import pytest

import elasticai.creator.graph as gr
import elasticai.creator.ir as ir
from elasticai.creator.ir.rewriting.rewriter import RemappedSubImplementation


class Node(ir.Node):
    size: int


class TestRemapping:
    @pytest.fixture
    def replaced(self) -> ir.Implementation:
        return (
            ir.Implementation(graph=gr.BaseGraph(), data={})
            .add_nodes(
                (
                    ir.node(name="a", type="ta"),
                    ir.node(name="b", type="tb"),
                    ir.node(name="c", type="tc"),
                )
            )
            .add_edges((ir.edge("a", "b"), ir.edge("b", "c"), ir.edge("a", "c")))
        )

    @pytest.fixture
    def replacement(self):
        return (
            ir.Implementation(graph=gr.BaseGraph(), data={})
            .add_nodes(
                (
                    ir.node(name="x", type="ta"),
                    ir.node(name="y", type="tb"),
                    ir.node(name="z", type="tc"),
                )
            )
            .add_edges((ir.edge("x", "y"), ir.edge("y", "z"), ir.edge("x", "z")))
        )

    @pytest.fixture
    def remapped(
        self,
        replaced: ir.Implementation[ir.Node, ir.Edge],
        replacement: ir.Implementation[ir.Node, ir.Edge],
    ):
        return RemappedSubImplementation(
            mapping={"x": "a", "y": "b", "z": "c"},
            graph=replacement.graph,
            data=replaced.data["nodes"],  #  type: ignore
            node_fn=Node,
        )

    def test_change_node_data_using_names_from_replacement(
        self, remapped: RemappedSubImplementation[Node], replaced: ir.Implementation
    ):
        replacement_node_name = "x"
        corresponding_node_name_in_resulting_graph = "a"
        remapped.nodes[replacement_node_name].size = 4
        assert (
            replaced.nodes[corresponding_node_name_in_resulting_graph].attributes[
                "size"
            ]
            == 4
        )
