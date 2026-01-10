from typing import cast

from torch import nn

from elasticai.creator.torch2ir import Implementation

handlers = []


def _register(fn):
    handlers.append(fn)
    return fn


@_register
def linear(impl: Implementation) -> nn.Module:
    if not hasattr(impl, "data"):
        attrs = impl.attributes
    else:
        attrs = impl.data
    return nn.Linear(
        in_features=cast(int, attrs["in_features"]),
        out_features=cast(int, attrs["out_features"]),
        bias=cast(bool, attrs["bias"]),
    )


@_register
def relu(impl: Implementation) -> nn.Module:
    return nn.ReLU()


@_register
def conv1d(impl: Implementation) -> nn.Conv1d:
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
def sigmoid(impl: Implementation) -> nn.Sigmoid:
    return nn.Sigmoid()


@_register
def flatten(imp: Implementation) -> nn.Flatten:
    return nn.Flatten()
