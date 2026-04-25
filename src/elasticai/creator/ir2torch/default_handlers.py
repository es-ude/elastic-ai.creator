from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn

import elasticai.creator.ir as ir
from elasticai.creator import function_dispatch as FD

dgraph_handlers = []
node_handlers = {}


def _register_dgraph(fn):
    dgraph_handlers.append(fn)
    return fn


@FD.registrar
def _register_node(
    key: str | None, fn: Callable[[ir.Node], Callable[[Any], torch.Tensor]]
):
    if key is None:
        if not hasattr(fn, "__name__"):
            raise TypeError("only functions are supported")
        key = cast(str, fn.__name__)
    node_handlers[key] = fn
    return fn


@_register_dgraph
def linear(impl: ir.DataGraph) -> nn.Module:
    if not hasattr(impl, "data"):
        attrs = impl.attributes
    return nn.Linear(
        in_features=cast(int, attrs["in_features"]),
        out_features=cast(int, attrs["out_features"]),
        bias=cast(bool, attrs["bias"]),
    )


@_register_dgraph
def relu(impl: ir.DataGraph) -> nn.Module:
    return nn.ReLU()


@_register_dgraph
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


@_register_dgraph
def sigmoid(impl: ir.DataGraph) -> nn.Sigmoid:
    return nn.Sigmoid()


@_register_dgraph
def flatten(imp: ir.DataGraph) -> nn.Flatten:
    return nn.Flatten(
        start_dim=imp.attributes["start_dim"], end_dim=imp.attributes["end_dim"]
    )


@_register_dgraph
def function(imp: ir.DataGraph) -> nn.Module:
    class FunctionWrapper(nn.Module):
        def __init__(self):
            self._fn = getattr(torch, imp.attributes["type"])

        def forward(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    return FunctionWrapper()


@_register_node("flatten")
@_register_node("add")
def handle_functions(node: ir.Node) -> Callable[[Any], torch.Tensor]:
    fn_name = node.attributes["implementation"]
    return getattr(torch, fn_name)
