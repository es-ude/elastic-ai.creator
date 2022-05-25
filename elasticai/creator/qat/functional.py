""" This file corresponds to the functional module in pytorch. We keep here custom autograd functions
"""

from typing import Any

import torch
from torch import Tensor, jit


@jit.script
def _heaviside(x):
    return torch.heaviside(x, torch.tensor([1.0]))


# noinspection PyPep8Naming,PyAbstractClass
class heaviside_with_ste(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 0:
            raise TypeError
        x = args[0]
        masked_zero_for_out_of_range = _heaviside(x + 1) - _heaviside(x - 1)
        ctx.save_for_backward(masked_zero_for_out_of_range)
        y = 2 * _heaviside(x) - 1
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (masked_zero_for_out_of_range_inputs,) = ctx.saved_tensors
        return grad_outputs[0] * masked_zero_for_out_of_range_inputs


binarize = heaviside_with_ste
