from collections.abc import Callable
from functools import wraps

import torch
from elasticai.creator_plugins.lutron_filter.nn.binarize import Binarize
from torch.nn import Module

import elasticai.creator.torch2ir as t2i
from elasticai.creator.ir import AttributeConvertable
from elasticai.creator.torch2ir.default_handlers import batchnorm1d as _eai_bnorm

type _Handler = Callable[[Module], dict[str, AttributeConvertable]]


def _extend_handler_with_parameter_names(handler: _Handler) -> _Handler:
    @wraps(handler)
    def wrapper(module: Module, /) -> dict[str, AttributeConvertable]:
        result = handler(module)
        return result | (
            {"parameters": dict((k, v.tolist()) for k, v in module._parameters.items())}  # type: ignore
        )

    return wrapper


_registered = list(
    map(_extend_handler_with_parameter_names, t2i.default_module_handlers)
)
_overriden: list[_Handler] = []


def _register(handler: _Handler) -> _Handler:
    _registered.append(handler)
    return handler


def _override(handler: _Handler) -> _Handler:
    _overriden.append(handler)
    return handler


@_register
def prelu(module: Module) -> dict[str, AttributeConvertable]:
    return {}


@_register
def binarize(module: Module) -> dict[str, AttributeConvertable]:
    return {}


@_override
def batchnorm1d(module: Module) -> dict[str, AttributeConvertable]:
    return {
        "running_mean": module.running_mean.tolist(),  # type: ignore
        "running_var": module.running_var.tolist(),  # type: ignore
    } | _eai_bnorm(module)


@_override
def flatten(module: Module) -> dict:
    return {"start_dim": module.start_dim, "end_dim": module.end_dim}


class _Tracer(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, Binarize):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def get_default_torch2ir() -> t2i.Torch2Ir:
    t = t2i.Torch2Ir(tracer=_Tracer())
    for h in _registered:
        t.register()(h)
    for h in _overriden:
        t.override()(h)

    return t
