from typing import Any, Callable

import torch
from torch.nn import Module

from elasticai.creator.tags_utils import get_tags


def _preprocess_tags(tags: dict):
    processed = {}
    unprocessed = []
    for key, value in tags.items():
        to_check = value
        if isinstance(value, list):
            to_check = value[0]
        if isinstance(to_check, int):
            processed[key + "_i"] = value
        elif isinstance(to_check, float):
            processed[key + "_f"] = value
        elif isinstance(to_check, str):
            processed[key + "_s"] = value
        elif isinstance(to_check, torch.Tensor):
            processed[key + "_t"] = value
        else:
            unprocessed.append(value)
    return processed, unprocessed


class AutogradWrapper(torch.autograd.Function):
    """
    Adds symbolic function so the tags can be exported. Is transparent

    """

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def forward(ctx: Any, input: Any, callable: Callable) -> Any:
        return callable(input)

    @staticmethod
    def symbolic(g, x, wrapped: Module):
        tags = get_tags(wrapped)
        kwargs, args = _preprocess_tags(tags)
        ret = g.op(
            "elasticai.creator::Wrapper",
            *args,
            operation_name_s=type(wrapped).__name__,
            **kwargs
        )
        return ret


class ModuleWrapper(Module):
    """
    Wraps the module so that it applies an autograd

    """

    def __init__(
        self,
        module,
        autograd_fn=AutogradWrapper,
    ):
        super().__init__()
        self.autograd_fn = autograd_fn
        self.module = module

    def forward(self, x):
        return self.autograd_fn.apply(x, self.module)
