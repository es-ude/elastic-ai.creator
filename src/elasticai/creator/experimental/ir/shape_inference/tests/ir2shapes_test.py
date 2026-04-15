import pytest
from torch.nn import BatchNorm2d, Conv2d, Flatten, Linear, Module, ReLU, Sequential

from elasticai.creator import ir, torch2ir
from elasticai.creator.experimental.ir.shape_inference import (
    get_default_shape_inference,
)
from elasticai.creator.torch2ir.torch2ir import DataGraph, Registry


def model():
    return Sequential(
        Linear(1, 2),
        ReLU(),
    )


def model_skip_connection():
    class SkipConnection(Module):
        def __init__(self):
            super().__init__()
            self.linear = Linear(72, 1)
            self.relu = ReLU()
            self.conv2d = Conv2d(2, 2, 3, padding="same")
            self.batchnorm2d = BatchNorm2d(2)
            self.flatten = Flatten()

        def forward(self, x):
            identity = x
            y = self.conv2d(x)
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
    assert serialize(new_graph, reg) == {
        "0": {
            "type": "linear",
            "in_features": 1,
            "out_features": 2,
            "bias": True,
            "nodes": {},
            "edges": {},
        },
        "1": {"type": "relu", "nodes": {}, "edges": {}},
        "": {
            "type": "module",
            "nodes": {
                "input_1": {"type": "input", "implementation": "input"},
                "_0": {"type": "linear", "implementation": "0"},
                "_1": {"type": "relu", "implementation": "1"},
                "output": {"type": "output", "implementation": "output"},
            },
            "edges": {
                "input_1": {"_0": {"shape": (5, 1)}},
                "_0": {"_1": {"shape": (5, 2)}},
                "_1": {"output": {"shape": (5, 2)}},
                "output": {},
            },
        },
    }


def test_ir2shapes_skip_connection(convert2ir):
    shape = (5, 2, 6, 6)
    m = model_skip_connection()
    root, reg = convert2ir(m)
    translator = shape_translator()
    new_graph = translator(root, reg, {"x": shape})
    assert serialize(new_graph, reg) == {
        "conv2d": {
            "type": "conv2d",
            "in_channels": 2,
            "out_channels": 2,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": "same",
            "dilation": (1, 1),
            "groups": 1,
            "bias": True,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "relu": {"type": "relu", "nodes": {}, "edges": {}},
        "batchnorm2d": {
            "type": "batchnorm2d",
            "num_features": 2,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "add": {"type": "add", "nodes": {}, "edges": {}},
        "flatten": {"type": "flatten", "nodes": {}, "edges": {}},
        "linear": {
            "type": "linear",
            "in_features": 72,
            "out_features": 1,
            "bias": True,
            "nodes": {},
            "edges": {},
        },
        "": {
            "type": "module",
            "nodes": {
                "x": {"type": "input", "implementation": "input"},
                "conv2d": {"type": "conv2d", "implementation": "conv2d"},
                "add": {"type": "add", "implementation": "add"},
                "relu": {"type": "relu", "implementation": "relu"},
                "batchnorm2d": {"type": "batchnorm2d", "implementation": "batchnorm2d"},
                "flatten": {"type": "flatten", "implementation": "flatten"},
                "linear": {"type": "linear", "implementation": "linear"},
                "relu_1": {"type": "relu", "implementation": "relu"},
                "output": {"type": "output", "implementation": "output"},
            },
            "edges": {
                "x": {
                    "conv2d": {"shape": (5, 2, 6, 6)},
                    "add": {"shape": (5, 2, 6, 6)},
                },
                "conv2d": {"relu": {"shape": (5, 2, 6, 6)}},
                "add": {"flatten": {"shape": (5, 2, 6, 6)}},
                "relu": {"batchnorm2d": {"shape": (5, 2, 6, 6)}},
                "batchnorm2d": {"add": {"shape": (5, 2, 6, 6)}},
                "flatten": {"linear": {"shape": (5, 72)}},
                "linear": {"relu_1": {"shape": (5, 1)}},
                "relu_1": {"output": {"shape": (5, 1)}},
                "output": {},
            },
        },
    }
