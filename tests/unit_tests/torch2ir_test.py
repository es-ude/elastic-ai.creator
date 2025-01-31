import torch
from torch.nn import Linear, ReLU, Sequential

from elasticai.creator.torch2ir import Torch2Ir


def model():
    return Sequential(
        Linear(1, 2),
        ReLU(),
    )


def convert(model):
    ir = Torch2Ir.get_default_converter().convert(model)
    return {impl.name: impl.data for impl in ir.values()}


def test_convert_linear_without_bias():
    m = Sequential(Linear(1, 2, bias=False))
    with torch.no_grad():
        m.get_submodule("0").weight.mul_(0).add_(1)
    ir = convert(m)
    assert ir == {
        "root": {
            "name": "root",
            "type": "module",
            "nodes": {
                "input_1": {
                    "name": "input_1",
                    "type": "input",
                    "implementation": "input",
                },
                "_0": {
                    "name": "_0",
                    "type": "linear",
                    "implementation": "0",
                },
                "output": {
                    "name": "output",
                    "type": "output",
                    "implementation": "output",
                },
            },
            "edges": {
                ("input_1", "_0"): {"src": "input_1", "sink": "_0"},
                ("_0", "output"): {"src": "_0", "sink": "output"},
            },
        },
        "0": {
            "name": "0",
            "type": "linear",
            "in_features": 1,
            "out_features": 2,
            "bias": False,
            "edges": {},
            "nodes": {},
        },
    }


def test_convert_linear_to_ir():
    m = model()
    with torch.no_grad():
        linear = m.get_submodule("0")
        linear.bias.mul_(0)
        linear.weight.mul_(0).add_(1)

    assert convert(m) == {
        "0": {
            "type": "linear",
            "in_features": 1,
            "out_features": 2,
            "bias": True,
            "edges": {},
            "name": "0",
            "nodes": {},
        },
        "1": {
            "type": "relu",
            "name": "1",
            "edges": {},
            "nodes": {},
        },
        "root": {
            "name": "root",
            "type": "module",
            "nodes": {
                "input_1": {
                    "implementation": "input",
                    "name": "input_1",
                    "type": "input",
                },
                "_0": {
                    "name": "_0",
                    "implementation": "0",
                    "type": "linear",
                },
                "_1": {
                    "implementation": "1",
                    "name": "_1",
                    "type": "relu",
                },
                "output": {
                    "implementation": "output",
                    "name": "output",
                    "type": "output",
                },
            },
            "edges": {
                ("input_1", "_0"): {"src": "input_1", "sink": "_0"},
                ("_0", "_1"): {"src": "_0", "sink": "_1"},
                ("_1", "output"): {"src": "_1", "sink": "output"},
            },
        },
    }


def test_converting_model_with_batchnorm():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(2)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(x))

    m = Model()
    assert convert(m) == {
        "bn": {
            "affine": True,
            "edges": {},
            "name": "bn",
            "nodes": {},
            "num_features": 2,
            "type": "batchnorm1d",
        },
        "relu": {
            "edges": {},
            "name": "relu",
            "nodes": {},
            "type": "relu",
        },
        "root": {
            "edges": {
                ("bn", "relu"): {
                    "sink": "relu",
                    "src": "bn",
                },
                ("relu", "output"): {
                    "sink": "output",
                    "src": "relu",
                },
                ("x", "bn"): {
                    "sink": "bn",
                    "src": "x",
                },
            },
            "name": "root",
            "nodes": {
                "bn": {
                    "implementation": "bn",
                    "name": "bn",
                    "type": "batchnorm1d",
                },
                "output": {
                    "implementation": "output",
                    "name": "output",
                    "type": "output",
                },
                "relu": {
                    "implementation": "relu",
                    "name": "relu",
                    "type": "relu",
                },
                "x": {
                    "implementation": "input",
                    "name": "x",
                    "type": "input",
                },
            },
            "type": "module",
        },
    }
