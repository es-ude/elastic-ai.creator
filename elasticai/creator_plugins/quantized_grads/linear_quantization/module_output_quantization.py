from doctest import OutputChecker

import torch
from torch import Tensor
from torch.nn import Module

from tests.unit_tests.base_modules.lstm_test import OutputsZeroLSTMCell
from .autograd import (
    QuantizeFakeLinearAsymForwHTEAutograd,
    QuantizeFakeLinearAsymForwStochasticAutograd,
)
from .linear_quantization_config import LinearQuantizationConfig, IntQuantizationConfig


class _OutputModule(Module):
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
        self.autograd_func = None
    def forward(self, x: Tensor) -> Tensor:...

def get_linear_quantization_forw_module(
        autograd_func: torch.autograd.Function,
) -> type[_OutputModule]:
    class LinearQuantizationOutputQuantizationModule(_OutputModule):
        def __init__(self, forward_conf: LinearQuantizationConfig) -> None:
            super().__init__(forward_conf)
            self.autograd_func = autograd_func

        def forward(self, x: Tensor) -> Tensor:
            return self.autograd_func.apply(
                x,
                self.forw_min_value,
                self.forw_max_value
            )

    return LinearQuantizationOutputQuantizationModule

OutputQuantizeLinearAsymForwHTE: type[_OutputModule] = get_linear_quantization_forw_module(
    QuantizeFakeLinearAsymForwHTEAutograd
)
OutputQuantizeLinearAsymForwStochastic: type[_OutputModule] = get_linear_quantization_forw_module(
    QuantizeFakeLinearAsymForwStochasticAutograd
)
