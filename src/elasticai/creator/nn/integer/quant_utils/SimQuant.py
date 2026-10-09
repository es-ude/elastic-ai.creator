from typing import Any

import torch
from torch.autograd import Function

from elasticai.creator.nn.integer._dtypes import FLOAT_DTYPES, check_dtype


class SimQuant(Function):
    """Simulated (fake) quantization with a straight-through estimator.

    The forward pass quantizes and dequantizes the input, so the output only
    takes values representable by the given quantization parameters. The
    backward pass passes gradients through unchanged inside the observed float
    range and sets them to zero outside of it.

    Use it via `SimQuant.apply(x_r, x_r_QParams)`.
    """

    @staticmethod
    def forward(ctx: Any, x_r: torch.Tensor, x_r_QParams: Any) -> torch.Tensor:
        """Quantize and dequantize `x_r`.

        Args:
            ctx: Autograd context used to save tensors for the backward pass.
            x_r: Float input tensor.
            x_r_QParams: The `QParams` object used for quantization.

        Returns:
            The dequantized tensor, with the same shape as `x_r`.

        Raises:
            TypeError: If `x_r` is not a `torch.Tensor` or does not have a
                floating-point dtype.
        """
        check_dtype("x_r", x_r, *FLOAT_DTYPES)
        x_q = x_r_QParams.quantize(x_r).to(x_r.device)
        min_float = x_r_QParams.min_float.to(x_r.device)
        max_float = x_r_QParams.max_float.to(x_r.device)

        # ctx.save_for_backward(x_r, min_float, max_float)
        ctx.save_for_backward(x_r.clone(), min_float.clone(), max_float.clone())

        return x_r_QParams.dequantize(x_q)

    # NOTE: ty reports invalid-method-override because the base
    #   `Function.backward(ctx, *grad_outputs)` accepts any number of gradients,
    #    while this override accepts exactly one. Autograd is the only caller and
    #   passes one gradient per output of `forward`, which returns one tensor.
    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        """Pass gradients through where the input lies inside the float range.

        Args:
            ctx: Autograd context holding the tensors saved in `forward`.
            grad_output: Gradient with respect to the output.

        Returns:
            The gradient with respect to `x_r`, zeroed where `x_r` lies outside
            `[min_float, max_float]`, followed by `None` for the other inputs.
        """
        x_r, min_float, max_float = ctx.saved_tensors
        grad_output = grad_output.to(x_r.device)
        grad_input = grad_output.clone()
        grad_input[x_r.gt(max_float)] = 0
        grad_input[x_r.lt(min_float)] = 0
        return grad_input, None, None, None
