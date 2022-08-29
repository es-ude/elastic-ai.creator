""" This file corresponds to the functional module in pytorch. We keep here custom autograd functions
"""

from typing import Any

import torch


class binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 0:
            raise TypeError
        x = args[0]
        out_of_range = torch.logical_or(torch.gt(x, 1.0), torch.lt(x, -1.0))
        ctx.save_for_backward(out_of_range)
        y = torch.where(x >= 0, 1.0, -1.0)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (out_of_range,) = ctx.saved_tensors
        return grad_outputs[0] * torch.where(out_of_range, 0.0, 1.0)
