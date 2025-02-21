from collections import OrderedDict

import torch
from torch.nn import Linear, ReLU, Sequential

from elasticai.creator.torch2ir import get_default_converter


def model():
    return Sequential(
        Linear(1, 2),
        ReLU(),
    )


def convert(model):
    ir = get_default_converter().convert(model)
    return [impl.as_dict() for impl in ir]


def test_convert_linear_without_bias():
    m = Sequential(Linear(1, 2, bias=False))
    with torch.no_grad():
        m.get_submodule("0").weight.mul_(0).add_(1)
    ir = convert(m)
    assert ir == [
        {
            "name": "",
            "type": "module",
            "nodes": [
                {
                    "name": "input_1",
                    "type": "input",
                    "implementation": "input",
                },
                {
                    "name": "_0",
                    "type": "linear",
                    "implementation": "0",
                },
                {
                    "name": "output",
                    "type": "output",
                    "implementation": "output",
                },
            ],
            "edges": [
                {"src": "input_1", "sink": "_0"},
                {"src": "_0", "sink": "output"},
            ],
        },
        {
            "name": "0",
            "type": "linear",
            "in_features": 1,
            "out_features": 2,
            "bias": False,
            "edges": [],
            "nodes": [],
        },
    ]


def test_convert_linear_to_ir():
    m = model()
    with torch.no_grad():
        linear = m.get_submodule("0")
        linear.bias.mul_(0)
        linear.weight.mul_(0).add_(1)

    assert convert(m) == [
        {
            "name": "",
            "type": "module",
            "nodes": [
                {
                    "implementation": "input",
                    "name": "input_1",
                    "type": "input",
                },
                {
                    "name": "_0",
                    "implementation": "0",
                    "type": "linear",
                },
                {
                    "implementation": "1",
                    "name": "_1",
                    "type": "relu",
                },
                {
                    "implementation": "output",
                    "name": "output",
                    "type": "output",
                },
            ],
            "edges": [
                {"src": "input_1", "sink": "_0"},
                {"src": "_0", "sink": "_1"},
                {"src": "_1", "sink": "output"},
            ],
        },
        {
            "type": "linear",
            "in_features": 1,
            "out_features": 2,
            "bias": True,
            "edges": [],
            "name": "0",
            "nodes": [],
        },
        {
            "type": "relu",
            "name": "1",
            "edges": [],
            "nodes": [],
        },
    ]


def test_converting_model_with_batchnorm():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(2)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(x))

    m = Model()
    assert convert(m) == [
        {
            "edges": [
                {
                    "sink": "bn",
                    "src": "x",
                },
                {
                    "sink": "relu",
                    "src": "bn",
                },
                {
                    "sink": "output",
                    "src": "relu",
                },
            ],
            "name": "",
            "nodes": [
                {
                    "implementation": "input",
                    "name": "x",
                    "type": "input",
                },
                {
                    "implementation": "bn",
                    "name": "bn",
                    "type": "batchnorm1d",
                },
                {
                    "implementation": "relu",
                    "name": "relu",
                    "type": "relu",
                },
                {
                    "implementation": "output",
                    "name": "output",
                    "type": "output",
                },
            ],
            "type": "module",
        },
        {
            "name": "bn",
            "type": "batchnorm1d",
            "edges": [],
            "nodes": [],
            "num_features": 2,
            "affine": True,
        },
        {
            "edges": [],
            "name": "relu",
            "nodes": [],
            "type": "relu",
        },
    ]


def test_can_handle_same_object_under_different_hierarchy_paths():
    lin = Linear(1, 1)
    model = Sequential(OrderedDict(a=lin, b=lin))
    assert convert(model) == [
        {
            "name": "",
            "type": "module",
            "nodes": [
                {
                    "name": "input_1",
                    "type": "input",
                    "implementation": "input",
                },
                {
                    "name": "a",
                    "type": "linear",
                    "implementation": "a",
                },
                {
                    "name": "a_1",
                    "type": "linear",
                    "implementation": "a",
                },
                {
                    "name": "output",
                    "type": "output",
                    "implementation": "output",
                },
            ],
            "edges": [
                {"src": "input_1", "sink": "a"},
                {"src": "a", "sink": "a_1"},
                {"src": "a_1", "sink": "output"},
            ],
        },
        {
            "name": "a",
            "type": "linear",
            "in_features": 1,
            "out_features": 1,
            "bias": True,
            "edges": [],
            "nodes": [],
        },
    ]
