""" This file corresponds to the functional module in pytorch. We keep here custom autograd functions
"""

from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function


# noinspection PyPep8Naming
class binarize(Function):
    """
    Implementation of binarization with straight-through-estimator (STE).
    A lot of other operations, e.g. Ternarization, Multi-Level Residual Binarization, etc. in the layers module
    are using this function.
    """

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        """At the time of writing the official pytorch documentation is seriously lacking on this function,
        therefore we decide to not implement this.
        """
        raise NotImplementedError

    @staticmethod
    def _unwrap_inputs(inputs: Any) -> Tensor:
        """ Using the _unwrap_inputs method allows the forward method to keep
            the signature of the autograd.Function superclass.
        """
        return inputs

    @staticmethod
    def forward(ctx: Any, *inputs: Any, **kwargs) -> Any:
        inputs = binarize._unwrap_inputs(*inputs)
        ctx.save_for_backward(inputs)
        return (2 * inputs.sign() + torch.ones_like(inputs)).clip_(-1, 1)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        inputs, = ctx.saved_tensors
        mask = inputs.less(-1).logical_or(inputs.greater(1))
        return grad_output * torch.ones_like(inputs).masked_fill_(mask, 0)
