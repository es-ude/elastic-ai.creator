import elasticai.creator_plugins.lutron_filter as lf
import pytest
from elasticai.creator_plugins.lutron_filter import remove_redundant_layers

from elasticai.creator.ir import attribute


@pytest.fixture
def ir_repr(request):
    layer_type = request.param
    return lf.build_sequential_ir(
        dict(
            main=attribute(),
            a=attribute(type="filter"),
            b=attribute(type="filter"),
            redundant=attribute(type=layer_type),
        ),
        sequence=("a", "redundant", "b"),
    )


@pytest.mark.parametrize("ir_repr", ("sigmoid",), indirect=True)
def test_dont_remove_sigmoid_in_middle(ir_repr):
    expected, _ = ir_repr
    result, _ = lf.remove_redundant_layers(*ir_repr)
    assert get_impl_edges(expected) == get_impl_edges(result)


@pytest.mark.parametrize("ir_repr", ("binarize", "flatten"), indirect=True)
def test_remove_non_sigmoid(ir_repr: tuple[lf.DataGraph, lf.Registry[lf.DataGraph]]):
    result, _ = lf.remove_redundant_layers(*ir_repr)
    expected = (
        ir_repr[0]
        .remove_edge("0", "1")
        .remove_edge("1", "2")
        .remove_node("1")
        .add_edge("0", "2")
    )

    assert get_impl_edges(expected) == get_impl_edges(result)


def test_remove_final_sigmoid():

    g, reg = lf.build_sequential_ir(
        dict(
            main=attribute(),
            a=attribute(type="filter"),
            b=attribute(type="filter"),
            redundant=attribute(type="sigmoid"),
        ),
        sequence=("a", "b", "redundant"),
    )
    expected = g.remove_edge("2", "output").remove_node("2").add_edge("1", "output")
    g, _ = remove_redundant_layers(g, reg)
    assert get_impl_edges(expected) == get_impl_edges(g)


def get_edge_nodes(g):
    for edge in g.edges.values():
        yield g.nodes[edge.src], g.nodes[edge.dst]


def get_impl_edges(g: lf.DataGraph) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for src, dst in get_edge_nodes(g):
        edges.add((src.implementation, dst.implementation))
    return edges
