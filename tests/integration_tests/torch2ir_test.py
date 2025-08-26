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
            "nodes": {
                "input_1": {
                    "type": "input",
                    "implementation": "input",
                },
                "_0": {
                    "type": "linear",
                    "implementation": "0",
                },
                "output": {
                    "type": "output",
                    "implementation": "output",
                },
            },
            "edges": {"input_1": {"_0": dict()}, "_0": {"output": dict()}},
        },
        {
            "name": "0",
            "type": "linear",
            "in_features": 1,
            "out_features": 2,
            "bias": False,
            "edges": {},
            "nodes": {},
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
            "nodes": {
                "input_1": {
                    "implementation": "input",
                    "type": "input",
                },
                "_0": {
                    "implementation": "0",
                    "type": "linear",
                },
                "_1": {
                    "implementation": "1",
                    "type": "relu",
                },
                "output": {
                    "implementation": "output",
                    "type": "output",
                },
            },
            "edges": {"input_1": {"_0": {}}, "_0": {"_1": {}}, "_1": {"output": {}}},
        },
        {
            "type": "linear",
            "in_features": 1,
            "out_features": 2,
            "bias": True,
            "edges": {},
            "name": "0",
            "nodes": {},
        },
        {
            "type": "relu",
            "name": "1",
            "edges": {},
            "nodes": {},
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
            "edges": {"x": {"bn": {}}, "bn": {"relu": {}}, "relu": {"output": {}}},
            "name": "",
            "nodes": {
                "x": {
                    "implementation": "input",
                    "type": "input",
                },
                "bn": {
                    "implementation": "bn",
                    "type": "batchnorm1d",
                },
                "relu": {
                    "implementation": "relu",
                    "type": "relu",
                },
                "output": {
                    "implementation": "output",
                    "type": "output",
                },
            },
            "type": "module",
        },
        {
            "name": "bn",
            "type": "batchnorm1d",
            "edges": {},
            "nodes": {},
            "num_features": 2,
            "affine": True,
        },
        {
            "edges": {},
            "name": "relu",
            "nodes": {},
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
            "nodes": {
                "input_1": {
                    "type": "input",
                    "implementation": "input",
                },
                "a": {
                    "type": "linear",
                    "implementation": "a",
                },
                "a_1": {
                    "type": "linear",
                    "implementation": "a",
                },
                "output": {
                    "type": "output",
                    "implementation": "output",
                },
            },
            "edges": {
                "input_1": {"a": {}},
                "a": {"a_1": {}},
                "a_1": {"output": {}},
            },
        },
        {
            "name": "a",
            "type": "linear",
            "in_features": 1,
            "out_features": 1,
            "bias": True,
            "edges": {},
            "nodes": {},
        },
    ]


def test_can_handle_nested_hierarchies():
    model = Sequential(OrderedDict(top=Sequential(OrderedDict(nested=Linear(1, 1)))))
    assert convert(model) == [
        {
            "name": "",
            "type": "module",
            "nodes": {
                "input_1": {
                    "type": "input",
                    "implementation": "input",
                },
                "top_nested": {
                    "type": "linear",
                    "implementation": "top.nested",
                },
                "output": {
                    "type": "output",
                    "implementation": "output",
                },
            },
            "edges": {"input_1": {"top_nested": {}}, "top_nested": {"output": {}}},
        },
        {
            "name": "top.nested",
            "type": "linear",
            "in_features": 1,
            "out_features": 1,
            "bias": True,
            "edges": {},
            "nodes": {},
        },
    ]
