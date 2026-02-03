from collections import OrderedDict

import pytest
import torch
from torch.nn import BatchNorm2d, Conv2d, Linear, Module, ReLU, Sequential
from torchvision.models import resnet18

import elasticai.creator.ir.ir_v2 as ir
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

        def forward(self, x):
            identity = x
            y = self.conv2d(x)
            y = self.relu(y)
            y = self.batchnorm2d(y)
            y += identity
            y = torch.flatten(y)
            y = self.linear(y)
            y = self.relu(y)
            return y

    return SkipConnection()


def big_model():
    return resnet18(pretrained=True)


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


def convert_new_with_params(model):
    translate = Torch2IrWithParamsTranslator()
    translate.register()(linear)
    translate.register()(relu)
    translate.register()(batchnorm1d)
    serializer = ir.IrSerializerLegacy()

    root, reg = translate(model)
    result = {k: serializer.serialize(reg[k]) for k in reg}
    result[""] = serializer.serialize(root)
    return result


def convert_big(model):
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


@pytest.mark.parametrize("convert", [convert_new_with_params])
def test_convert_new_with_params(convert):
    m = model()
    m[0].weight = torch.nn.Parameter(torch.ones_like(m[0].weight))
    m[0].bias = torch.nn.Parameter(torch.zeros_like(m[0].bias))
    ir = convert(m)
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
            "bias": [
                0.0,
                0.0,
            ],
            "weight": [
                [1.0],
                [
                    1.0,
                ],
            ],
            "edges": {},
            "nodes": {},
        },
        "1": {
            "type": "relu",
            "edges": {},
            "nodes": {},
        },
    }


@pytest.mark.parametrize("convert", [convert_big])
def test_convert_linear_model_to_ir(convert):
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


@pytest.mark.parametrize("convert", [convert_big])
def test_convert_skip_connection_model_to_ir(convert):
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


@pytest.mark.parametrize("convert", [convert_big])
def test_convert_resnet_to_ir(convert):
    m = big_model()
    ir = convert(m)
    assert ir == {
        "conv1": {
            "type": "conv2d",
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": (7, 7),
            "stride": (2, 2),
            "padding": (3, 3),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "bn1": {
            "type": "batchnorm2d",
            "num_features": 64,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "relu": {"type": "relu", "nodes": {}, "edges": {}},
        "maxpool": {
            "type": "maxpool2d",
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "nodes": {},
            "edges": {},
        },
        "layer1.0.conv1": {
            "type": "conv2d",
            "in_channels": 64,
            "out_channels": 64,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer1.0.bn1": {
            "type": "batchnorm2d",
            "num_features": 64,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer1.0.relu": {"type": "relu", "nodes": {}, "edges": {}},
        "layer1.0.conv2": {
            "type": "conv2d",
            "in_channels": 64,
            "out_channels": 64,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer1.0.bn2": {
            "type": "batchnorm2d",
            "num_features": 64,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "add": {"type": "add", "nodes": {}, "edges": {}},
        "layer1.1.conv1": {
            "type": "conv2d",
            "in_channels": 64,
            "out_channels": 64,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer1.1.bn1": {
            "type": "batchnorm2d",
            "num_features": 64,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer1.1.relu": {"type": "relu", "nodes": {}, "edges": {}},
        "layer1.1.conv2": {
            "type": "conv2d",
            "in_channels": 64,
            "out_channels": 64,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer1.1.bn2": {
            "type": "batchnorm2d",
            "num_features": 64,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer2.0.conv1": {
            "type": "conv2d",
            "in_channels": 64,
            "out_channels": 128,
            "kernel_size": (3, 3),
            "stride": (2, 2),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer2.0.bn1": {
            "type": "batchnorm2d",
            "num_features": 128,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer2.0.relu": {"type": "relu", "nodes": {}, "edges": {}},
        "layer2.0.conv2": {
            "type": "conv2d",
            "in_channels": 128,
            "out_channels": 128,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer2.0.bn2": {
            "type": "batchnorm2d",
            "num_features": 128,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer2.0.downsample.0": {
            "type": "conv2d",
            "in_channels": 64,
            "out_channels": 128,
            "kernel_size": (1, 1),
            "stride": (2, 2),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer2.0.downsample.1": {
            "type": "batchnorm2d",
            "num_features": 128,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer2.1.conv1": {
            "type": "conv2d",
            "in_channels": 128,
            "out_channels": 128,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer2.1.bn1": {
            "type": "batchnorm2d",
            "num_features": 128,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer2.1.relu": {"type": "relu", "nodes": {}, "edges": {}},
        "layer2.1.conv2": {
            "type": "conv2d",
            "in_channels": 128,
            "out_channels": 128,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer2.1.bn2": {
            "type": "batchnorm2d",
            "num_features": 128,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer3.0.conv1": {
            "type": "conv2d",
            "in_channels": 128,
            "out_channels": 256,
            "kernel_size": (3, 3),
            "stride": (2, 2),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer3.0.bn1": {
            "type": "batchnorm2d",
            "num_features": 256,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer3.0.relu": {"type": "relu", "nodes": {}, "edges": {}},
        "layer3.0.conv2": {
            "type": "conv2d",
            "in_channels": 256,
            "out_channels": 256,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer3.0.bn2": {
            "type": "batchnorm2d",
            "num_features": 256,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer3.0.downsample.0": {
            "type": "conv2d",
            "in_channels": 128,
            "out_channels": 256,
            "kernel_size": (1, 1),
            "stride": (2, 2),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer3.0.downsample.1": {
            "type": "batchnorm2d",
            "num_features": 256,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer3.1.conv1": {
            "type": "conv2d",
            "in_channels": 256,
            "out_channels": 256,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer3.1.bn1": {
            "type": "batchnorm2d",
            "num_features": 256,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer3.1.relu": {"type": "relu", "nodes": {}, "edges": {}},
        "layer3.1.conv2": {
            "type": "conv2d",
            "in_channels": 256,
            "out_channels": 256,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer3.1.bn2": {
            "type": "batchnorm2d",
            "num_features": 256,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer4.0.conv1": {
            "type": "conv2d",
            "in_channels": 256,
            "out_channels": 512,
            "kernel_size": (3, 3),
            "stride": (2, 2),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer4.0.bn1": {
            "type": "batchnorm2d",
            "num_features": 512,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer4.0.relu": {"type": "relu", "nodes": {}, "edges": {}},
        "layer4.0.conv2": {
            "type": "conv2d",
            "in_channels": 512,
            "out_channels": 512,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer4.0.bn2": {
            "type": "batchnorm2d",
            "num_features": 512,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer4.0.downsample.0": {
            "type": "conv2d",
            "in_channels": 256,
            "out_channels": 512,
            "kernel_size": (1, 1),
            "stride": (2, 2),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer4.0.downsample.1": {
            "type": "batchnorm2d",
            "num_features": 512,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer4.1.conv1": {
            "type": "conv2d",
            "in_channels": 512,
            "out_channels": 512,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer4.1.bn1": {
            "type": "batchnorm2d",
            "num_features": 512,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "layer4.1.relu": {"type": "relu", "nodes": {}, "edges": {}},
        "layer4.1.conv2": {
            "type": "conv2d",
            "in_channels": 512,
            "out_channels": 512,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": False,
            "padding_mode": "zeros",
            "nodes": {},
            "edges": {},
        },
        "layer4.1.bn2": {
            "type": "batchnorm2d",
            "num_features": 512,
            "affine": True,
            "nodes": {},
            "edges": {},
        },
        "avgpool": {
            "type": "adaptiveavgpool2d",
            "output_size": (1, 1),
            "nodes": {},
            "edges": {},
        },
        "flatten": {"type": "flatten", "nodes": {}, "edges": {}},
        "fc": {
            "type": "linear",
            "in_features": 512,
            "out_features": 1000,
            "bias": True,
            "nodes": {},
            "edges": {},
        },
        "": {
            "type": "module",
            "nodes": {
                "x": {"type": "input", "implementation": "input"},
                "conv1": {"type": "conv2d", "implementation": "conv1"},
                "bn1": {"type": "batchnorm2d", "implementation": "bn1"},
                "relu": {"type": "relu", "implementation": "relu"},
                "maxpool": {"type": "maxpool2d", "implementation": "maxpool"},
                "layer1_0_conv1": {
                    "type": "conv2d",
                    "implementation": "layer1.0.conv1",
                },
                "add": {"type": "add", "implementation": "add"},
                "layer1_0_bn1": {
                    "type": "batchnorm2d",
                    "implementation": "layer1.0.bn1",
                },
                "layer1_0_relu": {"type": "relu", "implementation": "layer1.0.relu"},
                "layer1_0_conv2": {
                    "type": "conv2d",
                    "implementation": "layer1.0.conv2",
                },
                "layer1_0_bn2": {
                    "type": "batchnorm2d",
                    "implementation": "layer1.0.bn2",
                },
                "layer1_0_relu_1": {"type": "relu", "implementation": "layer1.0.relu"},
                "layer1_1_conv1": {
                    "type": "conv2d",
                    "implementation": "layer1.1.conv1",
                },
                "add_1": {"type": "add", "implementation": "add"},
                "layer1_1_bn1": {
                    "type": "batchnorm2d",
                    "implementation": "layer1.1.bn1",
                },
                "layer1_1_relu": {"type": "relu", "implementation": "layer1.1.relu"},
                "layer1_1_conv2": {
                    "type": "conv2d",
                    "implementation": "layer1.1.conv2",
                },
                "layer1_1_bn2": {
                    "type": "batchnorm2d",
                    "implementation": "layer1.1.bn2",
                },
                "layer1_1_relu_1": {"type": "relu", "implementation": "layer1.1.relu"},
                "layer2_0_conv1": {
                    "type": "conv2d",
                    "implementation": "layer2.0.conv1",
                },
                "layer2_0_downsample_0": {
                    "type": "conv2d",
                    "implementation": "layer2.0.downsample.0",
                },
                "layer2_0_bn1": {
                    "type": "batchnorm2d",
                    "implementation": "layer2.0.bn1",
                },
                "layer2_0_relu": {"type": "relu", "implementation": "layer2.0.relu"},
                "layer2_0_conv2": {
                    "type": "conv2d",
                    "implementation": "layer2.0.conv2",
                },
                "layer2_0_bn2": {
                    "type": "batchnorm2d",
                    "implementation": "layer2.0.bn2",
                },
                "add_2": {"type": "add", "implementation": "add"},
                "layer2_0_downsample_1": {
                    "type": "batchnorm2d",
                    "implementation": "layer2.0.downsample.1",
                },
                "layer2_0_relu_1": {"type": "relu", "implementation": "layer2.0.relu"},
                "layer2_1_conv1": {
                    "type": "conv2d",
                    "implementation": "layer2.1.conv1",
                },
                "add_3": {"type": "add", "implementation": "add"},
                "layer2_1_bn1": {
                    "type": "batchnorm2d",
                    "implementation": "layer2.1.bn1",
                },
                "layer2_1_relu": {"type": "relu", "implementation": "layer2.1.relu"},
                "layer2_1_conv2": {
                    "type": "conv2d",
                    "implementation": "layer2.1.conv2",
                },
                "layer2_1_bn2": {
                    "type": "batchnorm2d",
                    "implementation": "layer2.1.bn2",
                },
                "layer2_1_relu_1": {"type": "relu", "implementation": "layer2.1.relu"},
                "layer3_0_conv1": {
                    "type": "conv2d",
                    "implementation": "layer3.0.conv1",
                },
                "layer3_0_downsample_0": {
                    "type": "conv2d",
                    "implementation": "layer3.0.downsample.0",
                },
                "layer3_0_bn1": {
                    "type": "batchnorm2d",
                    "implementation": "layer3.0.bn1",
                },
                "layer3_0_relu": {"type": "relu", "implementation": "layer3.0.relu"},
                "layer3_0_conv2": {
                    "type": "conv2d",
                    "implementation": "layer3.0.conv2",
                },
                "layer3_0_bn2": {
                    "type": "batchnorm2d",
                    "implementation": "layer3.0.bn2",
                },
                "add_4": {"type": "add", "implementation": "add"},
                "layer3_0_downsample_1": {
                    "type": "batchnorm2d",
                    "implementation": "layer3.0.downsample.1",
                },
                "layer3_0_relu_1": {"type": "relu", "implementation": "layer3.0.relu"},
                "layer3_1_conv1": {
                    "type": "conv2d",
                    "implementation": "layer3.1.conv1",
                },
                "add_5": {"type": "add", "implementation": "add"},
                "layer3_1_bn1": {
                    "type": "batchnorm2d",
                    "implementation": "layer3.1.bn1",
                },
                "layer3_1_relu": {"type": "relu", "implementation": "layer3.1.relu"},
                "layer3_1_conv2": {
                    "type": "conv2d",
                    "implementation": "layer3.1.conv2",
                },
                "layer3_1_bn2": {
                    "type": "batchnorm2d",
                    "implementation": "layer3.1.bn2",
                },
                "layer3_1_relu_1": {"type": "relu", "implementation": "layer3.1.relu"},
                "layer4_0_conv1": {
                    "type": "conv2d",
                    "implementation": "layer4.0.conv1",
                },
                "layer4_0_downsample_0": {
                    "type": "conv2d",
                    "implementation": "layer4.0.downsample.0",
                },
                "layer4_0_bn1": {
                    "type": "batchnorm2d",
                    "implementation": "layer4.0.bn1",
                },
                "layer4_0_relu": {"type": "relu", "implementation": "layer4.0.relu"},
                "layer4_0_conv2": {
                    "type": "conv2d",
                    "implementation": "layer4.0.conv2",
                },
                "layer4_0_bn2": {
                    "type": "batchnorm2d",
                    "implementation": "layer4.0.bn2",
                },
                "add_6": {"type": "add", "implementation": "add"},
                "layer4_0_downsample_1": {
                    "type": "batchnorm2d",
                    "implementation": "layer4.0.downsample.1",
                },
                "layer4_0_relu_1": {"type": "relu", "implementation": "layer4.0.relu"},
                "layer4_1_conv1": {
                    "type": "conv2d",
                    "implementation": "layer4.1.conv1",
                },
                "add_7": {"type": "add", "implementation": "add"},
                "layer4_1_bn1": {
                    "type": "batchnorm2d",
                    "implementation": "layer4.1.bn1",
                },
                "layer4_1_relu": {"type": "relu", "implementation": "layer4.1.relu"},
                "layer4_1_conv2": {
                    "type": "conv2d",
                    "implementation": "layer4.1.conv2",
                },
                "layer4_1_bn2": {
                    "type": "batchnorm2d",
                    "implementation": "layer4.1.bn2",
                },
                "layer4_1_relu_1": {"type": "relu", "implementation": "layer4.1.relu"},
                "avgpool": {"type": "adaptiveavgpool2d", "implementation": "avgpool"},
                "flatten": {"type": "flatten", "implementation": "flatten"},
                "fc": {"type": "linear", "implementation": "fc"},
                "output": {"type": "output", "implementation": "output"},
            },
            "edges": {
                "x": {"conv1": {}},
                "conv1": {"bn1": {}},
                "bn1": {"relu": {}},
                "relu": {"maxpool": {}},
                "maxpool": {"layer1_0_conv1": {}, "add": {}},
                "layer1_0_conv1": {"layer1_0_bn1": {}},
                "add": {"layer1_0_relu_1": {}},
                "layer1_0_bn1": {"layer1_0_relu": {}},
                "layer1_0_relu": {"layer1_0_conv2": {}},
                "layer1_0_conv2": {"layer1_0_bn2": {}},
                "layer1_0_bn2": {"add": {}},
                "layer1_0_relu_1": {"layer1_1_conv1": {}, "add_1": {}},
                "layer1_1_conv1": {"layer1_1_bn1": {}},
                "add_1": {"layer1_1_relu_1": {}},
                "layer1_1_bn1": {"layer1_1_relu": {}},
                "layer1_1_relu": {"layer1_1_conv2": {}},
                "layer1_1_conv2": {"layer1_1_bn2": {}},
                "layer1_1_bn2": {"add_1": {}},
                "layer1_1_relu_1": {"layer2_0_conv1": {}, "layer2_0_downsample_0": {}},
                "layer2_0_conv1": {"layer2_0_bn1": {}},
                "layer2_0_downsample_0": {"layer2_0_downsample_1": {}},
                "layer2_0_bn1": {"layer2_0_relu": {}},
                "layer2_0_relu": {"layer2_0_conv2": {}},
                "layer2_0_conv2": {"layer2_0_bn2": {}},
                "layer2_0_bn2": {"add_2": {}},
                "add_2": {"layer2_0_relu_1": {}},
                "layer2_0_downsample_1": {"add_2": {}},
                "layer2_0_relu_1": {"layer2_1_conv1": {}, "add_3": {}},
                "layer2_1_conv1": {"layer2_1_bn1": {}},
                "add_3": {"layer2_1_relu_1": {}},
                "layer2_1_bn1": {"layer2_1_relu": {}},
                "layer2_1_relu": {"layer2_1_conv2": {}},
                "layer2_1_conv2": {"layer2_1_bn2": {}},
                "layer2_1_bn2": {"add_3": {}},
                "layer2_1_relu_1": {"layer3_0_conv1": {}, "layer3_0_downsample_0": {}},
                "layer3_0_conv1": {"layer3_0_bn1": {}},
                "layer3_0_downsample_0": {"layer3_0_downsample_1": {}},
                "layer3_0_bn1": {"layer3_0_relu": {}},
                "layer3_0_relu": {"layer3_0_conv2": {}},
                "layer3_0_conv2": {"layer3_0_bn2": {}},
                "layer3_0_bn2": {"add_4": {}},
                "add_4": {"layer3_0_relu_1": {}},
                "layer3_0_downsample_1": {"add_4": {}},
                "layer3_0_relu_1": {"layer3_1_conv1": {}, "add_5": {}},
                "layer3_1_conv1": {"layer3_1_bn1": {}},
                "add_5": {"layer3_1_relu_1": {}},
                "layer3_1_bn1": {"layer3_1_relu": {}},
                "layer3_1_relu": {"layer3_1_conv2": {}},
                "layer3_1_conv2": {"layer3_1_bn2": {}},
                "layer3_1_bn2": {"add_5": {}},
                "layer3_1_relu_1": {"layer4_0_conv1": {}, "layer4_0_downsample_0": {}},
                "layer4_0_conv1": {"layer4_0_bn1": {}},
                "layer4_0_downsample_0": {"layer4_0_downsample_1": {}},
                "layer4_0_bn1": {"layer4_0_relu": {}},
                "layer4_0_relu": {"layer4_0_conv2": {}},
                "layer4_0_conv2": {"layer4_0_bn2": {}},
                "layer4_0_bn2": {"add_6": {}},
                "add_6": {"layer4_0_relu_1": {}},
                "layer4_0_downsample_1": {"add_6": {}},
                "layer4_0_relu_1": {"layer4_1_conv1": {}, "add_7": {}},
                "layer4_1_conv1": {"layer4_1_bn1": {}},
                "add_7": {"layer4_1_relu_1": {}},
                "layer4_1_bn1": {"layer4_1_relu": {}},
                "layer4_1_relu": {"layer4_1_conv2": {}},
                "layer4_1_conv2": {"layer4_1_bn2": {}},
                "layer4_1_bn2": {"add_7": {}},
                "layer4_1_relu_1": {"avgpool": {}},
                "avgpool": {"flatten": {}},
                "flatten": {"fc": {}},
                "fc": {"output": {}},
                "output": {},
            },
        },
    }


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
