from itertools import starmap

import pytest

from elasticai.creator.ir import attribute
from elasticai.creator.ir.datagraph import Node
from elasticai.creator.ir.datagraph_impl import DefaultIrFactory as IrFactory
from elasticai.creator.ir.registry import Registry
from elasticai.creator_plugins.lutron_filter.rules.binarize_activations import (
    binarize_activations as replace_prelu_with_binarize,
)


@pytest.fixture
def ir_factory():
    return IrFactory()


@pytest.fixture
def node(ir_factory):
    def create(name: str, type: str) -> Node:
        return ir_factory.node(name, attribute(type=type))

    return create


@pytest.fixture
def node_sequences(node) -> tuple[tuple[Node, ...], tuple[Node, ...]]:
    input = [
        ("input", "input"),
        ("a", "conv"),
        ("b", "conv"),
        ("c", "prelu"),
        ("output", "output"),
    ]
    expected = [
        ("input", "input"),
        ("a", "conv"),
        ("b", "conv"),
        ("activation", "binarize"),
        ("output", "output"),
    ]
    return tuple(starmap(node, input)), tuple(starmap(node, expected))


def graph_from_node_sequence(ir_factory, node, sequence):
    return (
        ir_factory.graph()
        .add_nodes(*sequence)
        .add_edges(
            *tuple((a.name, b.name) for a, b in zip(sequence[:-1], sequence[1:]))
        )
    )


@pytest.fixture
def expected(ir_factory, node, node_sequences):
    return graph_from_node_sequence(ir_factory, node, node_sequences[1])


@pytest.fixture
def original(ir_factory, node, node_sequences):
    return graph_from_node_sequence(ir_factory, node, node_sequences[0])


@pytest.fixture
def actual(original):
    _actual, _ = replace_prelu_with_binarize(original, Registry())
    return _actual


def test_successors_are_as_expected(actual, expected):
    assert dict(actual.successors) == dict(expected.successors)


def test_attributes_are_correct(actual, expected):
    assert actual.attributes == expected.attributes


def test_node_attributes_are_correct(actual, expected):
    assert dict(actual.node_attributes) == dict(expected.node_attributes)
