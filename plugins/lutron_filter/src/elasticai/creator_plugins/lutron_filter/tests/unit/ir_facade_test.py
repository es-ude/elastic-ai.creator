import pytest

import elasticai.creator_plugins.lutron_filter.rules._ir as _ir
from elasticai.creator.ir import attribute
from elasticai.creator.ir.datagraph_impl import DefaultIrFactory as Factory
from elasticai.creator.ir.serializer import IrSerializer


@pytest.fixture
def factory():
    return Factory()


_serializer = IrSerializer()


def serialize(g) -> dict:
    return _serializer.serialize(g)


def test_can_create_sequential(factory):
    expected = (
        factory.graph()
        .add_nodes(
            factory.node("a", attribute(type="ta")),
            factory.node("b", attribute(type="tb")),
            factory.node("c", attribute(type="tc")),
            factory.node("d", attribute(type="td")),
        )
        .add_edges(("a", "b"), ("b", "c"), ("c", "d"))
    )
    actual = _ir.sequential(("a", "ta"), ("b", "tb"), ("c", "tc"), ("d", "td"))
    assert serialize(actual) == serialize(expected)


def test_can_create_sequential_with_interface(factory):
    expected = (
        factory.graph()
        .add_nodes(
            factory.node("start", attribute(type="interface")),
            factory.node("b", attribute(type="tb")),
            factory.node("c", attribute(type="tc")),
            factory.node("end", attribute(type="interface")),
        )
        .add_edges(("start", "b"), ("b", "c"), ("c", "end"))
    )
    actual = _ir.sequential_with_interface(("b", "tb"), ("c", "tc"))
    assert serialize(actual) == serialize(expected)


def test_create_sequential_with_type_equal_name(factory):
    expected = (
        factory.graph()
        .add_nodes(_ir.node("a", "a"), _ir.node("b", "bt"))
        .add_edges(("a", "b"))
    )
    actual = _ir.sequential("a", ("b", "bt"))
    assert serialize(expected) == serialize(actual)


def test_build_sequential_ir_by_specifying_graph_implementation_sequence():
    graph, registry = _ir.build_sequential_ir(
        registry=dict(a=_ir.attribute(type="ta"), b=_ir.attribute(type="tb")),
        sequence=("a", "b", "a"),
    )
    nodes = set(graph.nodes)
    assert nodes == {"input", "output", "0", "1", "2"}
    edges = set(graph.edges)
    assert edges == {("0", "1"), ("input", "0"), ("1", "2"), ("2", "output")}
    assert graph.nodes["0"].implementation == "a"
    assert graph.nodes["0"].type == "ta"
