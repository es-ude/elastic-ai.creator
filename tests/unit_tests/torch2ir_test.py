from collections import OrderedDict

import pytest
import torch
from torch.nn import BatchNorm2d, Conv2d, Linear, Module, ReLU, Sequential, Flatten

from elasticai.creator import ir
from elasticai.creator.torch2ir import (
    Torch2Ir as Torch2IrTranslator,
)
from elasticai.creator.torch2ir import (
    Torch2IrWithParams as Torch2IrWithParamsTranslator,
)
from elasticai.creator.torch2ir.default_handlers import (
    adaptiveavgpool2d,
    add,
    batchnorm1d,
    batchnorm2d,
    conv2d,
    flatten,
    linear,
    maxpool2d,
    relu,
)


def model():
    return Sequential(
        Linear(1, 2),
        ReLU(),
    )


def model_skip_connection():
    class SkipConnection(Module):
        def __init__(self):
            super().__init__()
            self.linear = Linear(10, 1)
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


def convert_with_params(model):
    translate = Torch2IrWithParamsTranslator()
    translate.register()(flatten)
    translate.register()(linear)
    translate.register()(relu)
    translate.register()(batchnorm1d)
    translate.register()(conv2d)
    translate.register()(maxpool2d)
    translate.register()(batchnorm2d)
    translate.register()(adaptiveavgpool2d)
    translate.register()(add)
    serializer = ir.IrSerializerLegacy()

    root, reg = translate(model)
    result = {k: serializer.serialize(reg[k]) for k in reg}
    result[""] = serializer.serialize(root)
    return result


def convert(model):
    translate = Torch2IrTranslator()
    translate.register()(flatten)
    translate.register()(linear)
    translate.register()(relu)
    translate.register()(batchnorm1d)
    translate.register()(conv2d)
    translate.register()(maxpool2d)
    translate.register()(batchnorm2d)
    translate.register()(adaptiveavgpool2d)
    translate.register()(add)

    serializer = ir.IrSerializerLegacy()

    root, reg = translate(model)
    result = {k: serializer.serialize(reg[k]) for k in reg}
    result[""] = serializer.serialize(root)
    return result


def test_convert_big_with_params():
    m = model()
    m[0].weight = torch.nn.Parameter(torch.ones_like(m[0].weight))
    m[0].bias = torch.nn.Parameter(torch.zeros_like(m[0].bias))
    ir = convert_with_params(m)
    assert ir == {
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
            "bias": (
                0.0,
                0.0,
            ),
            "weight": (
                (1.0,),
                (1.0,),
            ),
            "edges": {},
            "nodes": {},
        },
        "1": {
            "type": "relu",
            "edges": {},
            "nodes": {},
        },
    }


def test_convert_linear_model_to_ir():
    m = model()
    ir = convert(m)
    assert ir == {
        "0": {
            "type": "linear",
            "in_features": 1,
            "out_features": 2,
            "bias": True,
            "nodes": {},
            "edges": {},
        },
        "1": {
            "type": "relu",
            "nodes": {},
            "edges": {},
        },
        "": {
            "type": "module",
            "nodes": {
                "input_1": {"type": "input", "implementation": "input"},
                "_0": {"type": "linear", "implementation": "0"},
                "_1": {"type": "relu", "implementation": "1"},
                "output": {"type": "output", "implementation": "output"},
            },
            "edges": {
                "input_1": {"_0": {}},
                "_0": {"_1": {}},
                "_1": {"output": {}},
                "output": {},
            },
        },
    }


def test_convert_skip_connection_model_to_ir():
    m = model_skip_connection()
    ir = convert(m)
    assert ir == {
        "add": {"edges": {}, "nodes": {}, "type": "add"},
        "batchnorm2d": {
            "affine": True,
            "edges": {},
            "nodes": {},
            "num_features": 2,
            "type": "batchnorm2d",
        },
        "conv2d": {
            "bias": True,
            "dilation": (1, 1),
            "edges": {},
            "groups": 1,
            "in_channels": 2,
            "kernel_size": (3, 3),
            "nodes": {},
            "out_channels": 2,
            "padding": "same",
            "padding_mode": "zeros",
            "stride": (1, 1),
            "type": "conv2d",
        },
        "flatten": {"edges": {}, "nodes": {}, "type": "flatten"},
        "linear": {
            "bias": True,
            "edges": {},
            "in_features": 10,
            "nodes": {},
            "out_features": 1,
            "type": "linear",
        },
        "relu": {"edges": {}, "nodes": {}, "type": "relu"},
        "": {
            "type": "module",
            "nodes": {
                "add": {"implementation": "add", "type": "add"},
                "batchnorm2d": {"implementation": "batchnorm2d", "type": "batchnorm2d"},
                "conv2d": {"implementation": "conv2d", "type": "conv2d"},
                "flatten": {"implementation": "flatten", "type": "flatten"},
                "linear": {"implementation": "linear", "type": "linear"},
                "output": {"implementation": "output", "type": "output"},
                "relu": {"implementation": "relu", "type": "relu"},
                "relu_1": {"implementation": "relu", "type": "relu"},
                "x": {"implementation": "input", "type": "input"},
            },
            "edges": {
                "add": {"flatten": {}},
                "batchnorm2d": {"add": {}},
                "conv2d": {"relu": {}},
                "flatten": {"linear": {}},
                "linear": {"relu_1": {}},
                "output": {},
                "relu": {"batchnorm2d": {}},
                "relu_1": {"output": {}},
                "x": {"add": {}, "conv2d": {}},
            },
        },
    }


def test_convert_linear_without_bias():
    m = Sequential(Linear(1, 2, bias=False))
    with torch.no_grad():
        m.get_submodule("0").weight.mul_(0).add_(1)  # ty:ignore[call-non-callable]
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


def test_convert_linear_to_ir():
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


def test_can_handle_same_object_under_different_hierarchy_paths():
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


def test_can_handle_nested_hierarchies():
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
