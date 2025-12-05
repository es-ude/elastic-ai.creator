import time
from typing import Callable

import torch
from torch import Tensor

from .quantize_linear import quantize_linear_asym_hte, quantize_linear_asym_stochastic, quantize_simulated_linear_asym_hte, \
    quantize_simulated_linear_asym_stochastic
from .quantize_to_int_with_linear_quantization_style import quantize_simulated_to_int_hte, quantize_simulated_to_int_stochastic


def get_autograd_func_quantisation(
        forw_simulated_quantisation_func: Callable[[Tensor, Tensor, Tensor], Tensor],
        forw_quantisation_func: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
) -> tuple[type[torch.autograd.Function], type[torch.autograd.Function]]:
    class QuantizationSimulatedForw(torch.autograd.Function):
        @staticmethod
        def forward(
                ctx,
                x: Tensor,
                min_value: Tensor,
                max_value: Tensor,
                *args,
                **kwargs,
        ):
            return forw_simulated_quantisation_func(
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

    return QuantizationSimulatedForw, QuantizationForw


QuantizeSimulatedLinearAsymForwHTEAutograd, QuantizeLinearAsymForwHTEAutograd = get_autograd_func_quantisation(quantize_simulated_linear_asym_hte, quantize_linear_asym_hte)

QuantizeSimulatedLinearAsymForwStochasticAutograd, QuantizeLinearAsymForwStochasticAutograd = get_autograd_func_quantisation(quantize_simulated_linear_asym_stochastic, quantize_linear_asym_stochastic)

QuantizeSimulatedIntForwHTEAutograd, _ = get_autograd_func_quantisation(quantize_simulated_to_int_hte, None)

QuantizeSimulatedIntForwStochasticAutograd, _ = get_autograd_func_quantisation(quantize_simulated_to_int_stochastic, None)