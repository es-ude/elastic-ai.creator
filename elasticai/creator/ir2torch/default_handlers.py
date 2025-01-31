from typing import cast

from torch import nn

from elasticai.creator.torch2ir import Implementation

handlers = []


def _register(fn):
    handlers.append(fn)
    return fn


@_register
def linear(impl: Implementation) -> nn.Module:
    return nn.Linear(
        in_features=cast(int, impl.data["in_features"]),
        out_features=cast(int, impl.data["out_features"]),
        bias=cast(bool, impl.data["bias"]),
    )


@_register
def relu(impl: Implementation) -> nn.Module:
    return nn.ReLU()
