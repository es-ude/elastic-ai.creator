from doctest import OutputChecker
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module

from tests.unit_tests.base_modules.lstm_test import OutputsZeroLSTMCell
from . import quantize_linear_asym_hte, quantize_simulated_linear_asym_hte, quantize_linear_asym_stochastic, \
    quantize_simulated_linear_asym_stochastic
from .autograd import (
    QuantizeSimulatedLinearAsymForwHTEAutograd,
    QuantizeSimulatedLinearAsymForwStochasticAutograd,
)
from .linear_quantization_config import LinearAsymQuantizationConfig, IntQuantizationConfig


class QuantizationModule(Module):
    def __init__(self, forward_conf: LinearAsymQuantizationConfig):
        super().__init__()
        self.register_buffer(
            "forw_min_value", forward_conf.min_value
        )
        self.register_buffer(
            "forw_max_value", forward_conf.max_value
        )

        self._kwargs = {
            "forw_min_value": self.forw_min_value,
            "forw_max_value": self.forw_max_value,
        }
        self.quantize_func = None
        self.quantize_simulated_func = None

    def forward(self, x: Tensor) -> Tensor:raise NotImplementedError
    def quantize(self, tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.quantize_func(tensor, self.forw_min_value, self.forw_max_value)
    def quantize_simulated(self, tensor: Tensor) -> Tensor:
        return self.quantize_simulated_func(tensor, self.forw_min_value, self.forw_max_value)

def get_linear_quantization_forw_module(
        quantize_func: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
        quantize_simulated_func: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> type[QuantizationModule]:
    class LinearQuantizationOutputQuantizationQuantizationModule(QuantizationModule):
        def __init__(self, forward_conf: LinearAsymQuantizationConfig) -> None:
            super().__init__(forward_conf)
            self.quantize_func = quantize_func
            self.quantize_simulated_func = quantize_simulated_func
    return LinearQuantizationOutputQuantizationQuantizationModule

ModuleQuantizeLinearAsymForwHTE: type[QuantizationModule] = get_linear_quantization_forw_module(
    quantize_linear_asym_hte, quantize_simulated_linear_asym_hte
)
ModuleQuantizeLinearAsymForwStochastic: type[QuantizationModule] = get_linear_quantization_forw_module(
    quantize_linear_asym_stochastic, quantize_simulated_linear_asym_stochastic
)
