import torch.nn as nn

handlers = []


def _register(fn):
    handlers.append(fn)
    return fn


@_register
def conv1d(module: nn.Conv1d) -> dict:
    return {
        "in_channels": module.in_channels,
        "out_channels": module.out_channels,
        "kernel_size": module.kernel_size,
        "stride": module.stride,
        "padding": module.padding,
        "dilation": module.dilation,
        "groups": module.groups,
        "bias": module.bias is not None,
        "padding_mode": module.padding_mode,
    }


@_register
def maxpool1d(module: nn.MaxPool1d) -> dict:
    return {
        "kernel_size": module.kernel_size,
        "stride": module.stride,
        "padding": module.padding,
        "dilation": module.dilation,
        "return_indices": module.return_indices,
        "ceil_mode": module.ceil_mode,
    }


@_register
def linear(module: nn.Linear) -> dict:
    return {
        "in_features": module.in_features,
        "out_features": module.out_features,
        "bias": module.bias is not None,
    }


@_register
def batchnorm1d(module: nn.BatchNorm1d) -> dict:
    return {
        "num_features": module.num_features,
        "affine": module.affine,
    }


@_register
def flatten(module: nn.Flatten) -> dict:
    return {}


@_register
def relu(module: nn.ReLU) -> dict:
    return {}


@_register
def sigmoid(module: nn.Sigmoid) -> dict:
    return {}
