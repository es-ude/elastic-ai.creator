from collections.abc import Callable

import torch.nn as nn

handlers: list[Callable[[nn.Module], dict]] = []


def _register(fn: Callable[[nn.Module], dict]) -> Callable[[nn.Module], dict]:
    handlers.append(fn)
    return fn


@_register
def conv1d(module: nn.Module) -> dict:
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
def maxpool1d(module: nn.Module) -> dict:
    return {
        "kernel_size": module.kernel_size,
        "stride": module.stride,
        "padding": module.padding,
        "dilation": module.dilation,
        "return_indices": module.return_indices,
        "ceil_mode": module.ceil_mode,
    }


@_register
def linear(module: nn.Module) -> dict:
    return {
        "in_features": module.in_features,
        "out_features": module.out_features,
        "bias": module.bias is not None,
    }


@_register
def batchnorm1d(module: nn.Module) -> dict:
    return {
        "num_features": module.num_features,
        "affine": module.affine,
    }


@_register
def flatten(module: nn.Module) -> dict:
    return {"start_dim": module.start_dim, "end_dim": module.end_dim}


@_register
def relu(module: nn.Module) -> dict:
    return {}


@_register
def sigmoid(module: nn.Module) -> dict:
    return {}

@_register
def prelu(_: nn.Module) -> dict:
    return {}

@_register
def conv2d(module: nn.Module) -> dict:
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
def batchnorm2d(module: nn.Module) -> dict:
    return {
        "num_features": module.num_features,
        "affine": module.affine,
    }


@_register
def maxpool2d(module: nn.Module) -> dict:
    return {
        "kernel_size": module.kernel_size,
        "stride": module.stride,
        "padding": module.padding,
        "dilation": module.dilation,
    }


@_register
def adaptiveavgpool2d(module: nn.Module) -> dict:
    return {
        "output_size": module.output_size,
    }


@_register
def add(module) -> dict:
    return {}
