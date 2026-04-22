from typing import cast

from torch import nn

import elasticai.creator.ir as ir

handlers = []


def _register(fn):
    handlers.append(fn)
    return fn


@_register
def linear(impl: ir.DataGraph) -> nn.Module:
    if not hasattr(impl, "data"):
        attrs = impl.attributes
    return nn.Linear(
        in_features=cast(int, attrs["in_features"]),
        out_features=cast(int, attrs["out_features"]),
        bias=cast(bool, attrs["bias"]),
    )


@_register
def relu(impl: ir.DataGraph) -> nn.Module:
    return nn.ReLU()


@_register
def conv1d(impl: ir.DataGraph) -> nn.Conv1d:
    keywords = (
        "in_channels",
        "out_channels",
        "bias",
        "groups",
        "kernel_size",
        "stride",
    )

    def get_attribute(impl, k):
        if hasattr(impl, "data"):
            return impl.data[k]
        return impl.attributes[k]

    kwargs = {k: get_attribute(impl, k) for k in keywords}
    return nn.Conv1d(**kwargs)  # type: ignore


@_register
def sigmoid(impl: ir.DataGraph) -> nn.Sigmoid:
    return nn.Sigmoid()


@_register
def flatten(imp: ir.DataGraph) -> nn.Flatten:
    return nn.Flatten()
