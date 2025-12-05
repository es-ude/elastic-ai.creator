from doctest import OutputChecker
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module

from tests.unit_tests.base_modules.lstm_test import OutputsZeroLSTMCell
from . import quantize_linear_asym_hte, quantize_linear_asym_hte_fake, quantize_linear_asym_stochastic, \
    quantize_linear_asym_stochastic_fake
from .autograd import (
    QuantizeFakeLinearAsymForwHTEAutograd,
    QuantizeFakeLinearAsymForwStochasticAutograd,
)
from .linear_quantization_config import LinearQuantizationConfig, IntQuantizationConfig


class _Module(Module):
    def __init__(self, forward_conf: LinearQuantizationConfig):
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
        self.quantize_fake_func = None

    def forward(self, x: Tensor) -> Tensor:raise NotImplementedError
    def quantize(self, tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.quantize_func(tensor, self.min_value, self.max_value)
    def quantize_fake(self, tensor: Tensor) -> Tensor:
        return self.quantize_fake_func(tensor, self.min_value, self.max_value)

def get_linear_quantization_forw_module(
        quantize_func: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
        quantize_fake_func: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> type[_Module]:
    class LinearQuantizationOutputQuantizationModule(_Module):
        def __init__(self, forward_conf: LinearQuantizationConfig) -> None:
            super().__init__(forward_conf)
            self.quantize_func = quantize_func
            self.quantize_fake_func = quantize_fake_func
    return LinearQuantizationOutputQuantizationModule

ModuleQuantizeLinearAsymForwHTE: type[_Module] = get_linear_quantization_forw_module(
    quantize_linear_asym_hte, quantize_linear_asym_hte_fake
)
ModuleQuantizeLinearAsymForwStochastic: type[_Module] = get_linear_quantization_forw_module(
    quantize_linear_asym_stochastic, quantize_linear_asym_stochastic_fake
)
