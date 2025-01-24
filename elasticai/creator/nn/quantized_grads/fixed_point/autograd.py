from typing import Callable

import torch
from torch import Tensor

from .quantize_to_fixed_point import quantize_to_fxp_hte, quantize_to_fxp_stochastic


def get_autograd_func(
    forw_func: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    backw_func: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
) -> (tuple)[type[torch.autograd.Function], type[torch.autograd.Function]]:
    class FixedPointForw(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            x: Tensor,
            forw_resolution_per_int: Tensor,
            forw_minimum_as_rational: Tensor,
            forw_maximum_as_rational: Tensor,
            *args,
            **kwargs,
        ):
            return forw_func(
                x,
                forw_resolution_per_int,
                forw_minimum_as_rational,
                forw_maximum_as_rational,
            )

        @staticmethod
        def backward(ctx, *grad_output):
            return *grad_output, None, None, None

    class FixedPointForwBackw(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            x: Tensor,
            forw_resolution_per_int: Tensor,
            forw_minimum_as_rational: Tensor,
            forw_maximum_as_rational: Tensor,
            backw_resolution_per_int: Tensor,
            backw_minimum_as_rational: Tensor,
            backw_maximum_as_rational: Tensor,
            *args,
            **kwargs,
        ):
            ctx.save_for_backward(
                backw_resolution_per_int,
                backw_minimum_as_rational,
                backw_maximum_as_rational,
            )
            return forw_func(
                x,
                forw_resolution_per_int,
                forw_minimum_as_rational,
                forw_maximum_as_rational,
            )

        @staticmethod
        def backward(ctx, *grad_output):
            (
                backw_resolution_per_int,
                backw_minimum_as_rational,
                backw_maximum_as_rational,
            ) = ctx.saved_tensors
            return (
                backw_func(
                    *grad_output,
                    backw_resolution_per_int,
                    backw_minimum_as_rational,
                    backw_maximum_as_rational,
                ),
                None,
                None,
                None,
                None,
                None,
                None,
            )

    return FixedPointForw, FixedPointForwBackw


QuantizeForwHTEAutograd, QuantizeForwHTEBackwHTEAutograd = get_autograd_func(
    quantize_to_fxp_hte, quantize_to_fxp_hte
)
(QuantizeForwStochasticAutograd, QuantizeForwStochasticBackwStochasticAutograd) = (
    get_autograd_func(quantize_to_fxp_stochastic, quantize_to_fxp_stochastic)
)

_, QuantizeForwHTEBackwStochasticAutograd = get_autograd_func(
    quantize_to_fxp_hte, quantize_to_fxp_stochastic
)
