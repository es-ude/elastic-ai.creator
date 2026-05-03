from typing import final, override

import pytest
from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, Flatten, Linear, Module, ReLU, Sequential

from elasticai.creator import ir, torch2ir
from elasticai.creator.experimental.ir.shape_inference import (
    get_default_shape_inference,
)
from elasticai.creator.ir import Registry
from elasticai.creator.torch2ir.torch2ir import DataGraph


def model():
    return Sequential(
        Linear(1, 2),
        ReLU(),
    )


def model_skip_connection():
    @final
    class SkipConnection(Module):
        def __init__(self):
            super().__init__()
            self.linear = Linear(72, 1)
            self.relu = ReLU()
            self.conv2d = Conv2d(2, 2, 3, padding="same")
            self.batchnorm2d = BatchNorm2d(2)
            self.flatten = Flatten()

        @override
        def forward(self, x: Tensor) -> Tensor:
            identity = x
            y: Tensor = self.conv2d(x)
            y = self.relu(y)
            y = self.batchnorm2d(y)
            y += identity
            y = self.flatten(y)
            y = self.linear(y)
            y = self.relu(y)
            return y

    return SkipConnection()


@pytest.fixture
def convert2ir() -> torch2ir.Torch2Ir:
    return torch2ir.get_default_converter()


def serialize(root: DataGraph, reg: Registry):
    serializer = ir.IrSerializerLegacy()
    result = {k: serializer.serialize(reg[k]) for k in reg}
    result[""] = serializer.serialize(root)
    return result


def shape_translator():
    return get_default_shape_inference()


def test_ir2shapes(convert2ir):
    m = model()
    root, reg = convert2ir(m)
    translator = shape_translator()
    new_graph = translator(root, reg, {"input_1": (5, 1)})

    expected_edges = {
        ("input_1", "_0", (5, 1)),
        ("_0", "_1", (5, 2)),
        ("_1", "output", (5, 2)),
    }
    actual_edges = {(e.src, e.dst, e.shape) for e in new_graph.edges.values()}
    assert expected_edges == actual_edges


def test_ir2shapes_skip_connection(convert2ir):
    input_shape = (5, 2, 6, 6)
    m = model_skip_connection()
    root, reg = convert2ir(m)
    translator = shape_translator()
    new_graph = translator(root, reg, {"x": input_shape})
    expected_shapes = {
        (a, b): s
        for a, b, s in (
            ("x", "conv2d", input_shape),
            ("x", "add", input_shape),
            ("conv2d", "relu", input_shape),
            ("batchnorm2d", "add", input_shape),
            ("add", "flatten", input_shape),
            ("relu", "batchnorm2d", input_shape),
            ("flatten", "linear", (5, 72)),
            ("linear", "relu_1", (5, 1)),
            ("relu_1", "output", (5, 1)),
        )
    }
    actual_shapes = {(e.src, e.dst): e.shape for e in new_graph.edges.values()}
    assert expected_shapes == actual_shapes
