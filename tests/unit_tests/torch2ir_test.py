from collections import OrderedDict

import pytest
import torch
from torch.nn import Linear, ReLU, Sequential

from elasticai.creator import ir
from elasticai.creator.torch2ir import (
    Torch2Ir as Torch2IrTranslator,
)
from elasticai.creator.torch2ir.default_handlers import batchnorm1d, linear, relu


def model():
    return Sequential(
        Linear(1, 2),
        ReLU(),
    )


def convert_new(model):
    translate = Torch2IrTranslator()
    translate.register()(linear)
    translate.register()(relu)
    translate.register()(batchnorm1d)
    serializer = ir.IrSerializerLegacy()

    root, reg = translate(model)
    result = {k: serializer.serialize(reg[k]) for k in reg}
    result[""] = serializer.serialize(root)
    return result


@pytest.mark.parametrize("convert", [convert_new])
def test_convert_linear_without_bias(convert):
    m = Sequential(Linear(1, 2, bias=False))
    with torch.no_grad():
        m.get_submodule("0").weight.mul_(0).add_(1)
    ir = convert(m)
    assert ir == {
        "": {
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
            "edges": {
                "input_1": {"_0": dict()},
                "_0": {"output": dict()},
                "output": {},
            },
        },
        "0": {
            "type": "linear",
            "in_features": 1,
            "out_features": 2,
            "bias": False,
            "edges": {},
            "nodes": {},
        },
    }


@pytest.mark.parametrize("convert", [convert_new])
def test_convert_linear_to_ir(convert):
    m = model()
    with torch.no_grad():
        linear = m.get_submodule("0")
        linear.bias.mul_(0)  # type: ignore
        linear.weight.mul_(0).add_(1)  # type: ignore

    assert convert(m) == {
        "": {
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
            "edges": {
                "input_1": {"_0": {}},
                "_0": {"_1": {}},
                "_1": {"output": {}},
                "output": {},
            },
        },
        "0": {
            "type": "linear",
            "in_features": 1,
            "out_features": 2,
            "bias": True,
            "edges": {},
            "nodes": {},
        },
        "1": {
            "type": "relu",
            "edges": {},
            "nodes": {},
        },
    }


@pytest.mark.parametrize("convert", [convert_new])
def test_converting_model_with_batchnorm(convert):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(2)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(x))

    m = Model()
    assert convert(m) == {
        "": {
            "edges": {
                "x": {"bn": {}},
                "bn": {"relu": {}},
                "relu": {"output": {}},
                "output": {},
            },
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
        "bn": {
            "type": "batchnorm1d",
            "edges": {},
            "nodes": {},
            "num_features": 2,
            "affine": True,
        },
        "relu": {
            "edges": {},
            "nodes": {},
            "type": "relu",
        },
    }


@pytest.mark.parametrize("convert", [convert_new])
def test_can_handle_same_object_under_different_hierarchy_paths(convert):
    lin = Linear(1, 1)
    model = Sequential(OrderedDict(a=lin, b=lin))  # type: ignore

    assert convert(model) == {
        "": {
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
                "output": {},
            },
        },
        "a": {
            "type": "linear",
            "in_features": 1,
            "out_features": 1,
            "bias": True,
            "edges": {},
            "nodes": {},
        },
    }


@pytest.mark.parametrize("convert", [convert_new])
def test_can_handle_nested_hierarchies(convert):
    model = Sequential(OrderedDict(top=Sequential(OrderedDict(nested=Linear(1, 1)))))  # type: ignore
    assert convert(model) == {
        "": {
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
            "edges": {
                "input_1": {"top_nested": {}},
                "top_nested": {"output": {}},
                "output": {},
            },
        },
        "top.nested": {
            "type": "linear",
            "in_features": 1,
            "out_features": 1,
            "bias": True,
            "edges": {},
            "nodes": {},
        },
    }
