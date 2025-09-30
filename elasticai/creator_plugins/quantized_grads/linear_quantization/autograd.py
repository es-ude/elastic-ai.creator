import time
from typing import Callable

import torch
from torch import Tensor

from .quantize_linear import quantize_linear_hte, quantize_linear_stochastic, quantize_linear_hte_fake, \
    quantize_linear_stochastic_fake

def get_autograd_func(
        forw_func: Callable[[Tensor, Tensor, Tensor], Tensor],
        backw_func: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> (tuple)[type[torch.autograd.Function], type[torch.autograd.Function]]:
    class LinearQuantizationForw(torch.autograd.Function):
        @staticmethod
        def forward(
                ctx,
                x: Tensor,
                min_value: Tensor,
                max_value: Tensor,
                *args,
                **kwargs,
        ):
            return forw_func(
                x,
                min_value,
                max_value,
            )

        @staticmethod
        def backward(ctx, *grad_output):
            return *grad_output, None, None

    class LinearQuantizationForwBackw(torch.autograd.Function):
        @staticmethod
        def forward(
                ctx,
                x: Tensor,
                forw_min_value: Tensor,
                forw_max_value: Tensor,
                backw_min_value: Tensor,
                backw_max_value: Tensor,
                *args,
                **kwargs,
        ):
            ctx.save_for_backward(
                backw_min_value,
                backw_max_value,
            )
            return forw_func(
                x,
                forw_min_value,
                forw_max_value,
            )

        @staticmethod
        def backward(ctx, *grad_output):
            (   backw_min_value,
                backw_max_value,
            ) = ctx.saved_tensors

            out = backw_func(
                *grad_output,
                backw_min_value,
                backw_max_value,
            )
            print(f"{grad_output=}")
            print(f"{out=}")
            return (
                out,
                None,
                None,
                None,
                None,
            )

    return LinearQuantizationForw, LinearQuantizationForwBackw


QuantizeForwHTEAutograd, QuantizeForwHTEBackwHTEAutograd = get_autograd_func(
    quantize_linear_hte_fake, quantize_linear_hte_fake
)
(QuantizeForwStochasticAutograd, QuantizeForwStochasticBackwStochasticAutograd) = (
    get_autograd_func(quantize_linear_stochastic_fake, quantize_linear_stochastic_fake)
)

_, QuantizeForwHTEBackwStochasticAutograd = get_autograd_func(
    quantize_linear_hte_fake, quantize_linear_stochastic_fake
)
