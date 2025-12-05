import time
from typing import Callable

import torch
from torch import Tensor

from .quantize_linear import quantize_linear_asym_hte, quantize_linear_asym_stochastic, quantize_linear_asym_hte_fake, \
    quantize_linear_asym_stochastic_fake
from .quantize_to_int_with_linear_quantization_style import quantize_to_int_hte_fake, quantize_to_int_stochastic_fake


def get_autograd_func_quantisation(
        forw_fake_quantisation_func: Callable[[Tensor, Tensor, Tensor], Tensor],
        forw_quantisation_func: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
) -> tuple[type[torch.autograd.Function], type[torch.autograd.Function]]:
    class QuantizationFakeForw(torch.autograd.Function):
        @staticmethod
        def forward(
                ctx,
                x: Tensor,
                min_value: Tensor,
                max_value: Tensor,
                *args,
                **kwargs,
        ):
            return forw_fake_quantisation_func(
                x,
                min_value,
                max_value,
            )

        @staticmethod
        def backward(ctx, *grad_output):
            return *grad_output, None, None

    class QuantizationForw(torch.autograd.Function):
        @staticmethod
        def forward(
                ctx,
                x: Tensor,
                min_value: Tensor,
                max_value: Tensor,
                *args,
                **kwargs,
        ):
            return forw_quantisation_func(
                x,
                min_value,
                max_value,
            )
        @staticmethod
        def backward(ctx, *grad_output):
            return *grad_output, None, None

    return QuantizationFakeForw, QuantizationForw


QuantizeFakeLinearAsymForwHTEAutograd, QuantizeLinearAsymForwHTEAutograd = get_autograd_func_quantisation(quantize_linear_asym_hte_fake, quantize_linear_asym_hte)

QuantizeFakeLinearAsymForwStochasticAutograd, QuantizeLinearAsymForwStochasticAutograd = get_autograd_func_quantisation(quantize_linear_asym_stochastic_fake, quantize_linear_asym_stochastic)

QuantizeFakeIntForwHTEAutograd, _ = get_autograd_func_quantisation(quantize_to_int_hte_fake, None)

QuantizeFakeIntForwStochasticAutograd, _ = get_autograd_func_quantisation(quantize_to_int_stochastic_fake, None)