import pytest

from elasticai.creator import ir
from elasticai.creator.experimental.ir.shape_inference import (
    IrShapeInference,
    Shape,
    get_default_shape_inference,
)

factory = ir.DefaultIrFactory()


@pytest.fixture
def infer_shape():
    infer = get_default_shape_inference()

    @infer.register_type()
    def scalar_function(input_shape: tuple[Shape, ...]) -> Shape:
        if len(input_shape) > 1:
            raise ValueError("Scalar function cannot take more than single argument")
        return input_shape[0]

    return infer


@pytest.fixture
def _base_graph():
    return factory.graph(ir.attribute(type="module")).add_nodes(
        factory.node("input", ir.attribute(type="input")),
        factory.node("output", ir.attribute(type="output")),
    )


@pytest.fixture
def _graph_with_datapath_join(_base_graph: ir.DataGraph[ir.Node, ir.Edge]):
    add_attr = ir.attribute(type="add")
    scalar_attr = ir.attribute(type="scalar_function")
    return (
        _base_graph.with_attributes(ir.attribute(name="child_module"))
        .add_nodes(
            ("a", scalar_attr),
            ("b", scalar_attr),
            ("c", add_attr),
        )
        .add_edges(
            ("input", "a"), ("input", "b"), ("a", "c"), ("b", "c"), ("c", "output")
        )
    )


def test_can_infer_shape_for_dgraph_with_path_join(
    _graph_with_datapath_join, infer_shape: IrShapeInference
):
    result = infer_shape(
        _graph_with_datapath_join,
        ir.Registry(),
        input_node_shapes=dict(input=(1, 1, 1)),
    )
    expected_edges = {
        (src, dst, (1, 1, 1))
        for src, dst in (
            ("input", "a"),
            ("input", "b"),
            ("a", "c"),
            ("b", "c"),
            ("c", "output"),
        )
    }
    actual_edges = {(e.src, e.dst, e.shape) for e in result.edges.values()}
    assert actual_edges == expected_edges
