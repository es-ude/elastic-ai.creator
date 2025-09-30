import torch
from torch import Tensor
from torch.nn import Module

from .autograd import (
    QuantizeForwHTEAutograd,
    QuantizeForwHTEBackwHTEAutograd,
    QuantizeForwHTEBackwStochasticAutograd,
    QuantizeForwStochasticAutograd,
    QuantizeForwStochasticBackwStochasticAutograd,
)
from .linear_quantization_config import LinearQuantizationConfig, IntQuantizationConfig


class _ForwModule(Module):
    def __init__(self, forward_conf: LinearQuantizationConfig): ...


class _ForwBackwModule(Module):
    def __init__(
            self, forward_conf: LinearQuantizationConfig, backward_conf: LinearQuantizationConfig
    ): ...


def get_linear_quantization_forw_module(
        autograd_func: torch.autograd.Function,
) -> type[_ForwModule]:
    class LinearQuantizationForwQuantizationModule(Module):
        def __init__(self, forward_conf: LinearQuantizationConfig | IntQuantizationConfig) -> None:
            super().__init__()
            self.autograd_func = autograd_func
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
        def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            return self.autograd_func.apply(
                x,
                self.forw_min_value,
                self.forw_max_value
            )
    return LinearQuantizationForwQuantizationModule


def get_linear_quantization_forwbackw_module(
        autograd_func: torch.autograd.Function,
) -> type[_ForwBackwModule]:
    class LinearQuantizationForwBackwQuantizationModule(Module):
        def __init__(
                self, forward_conf: LinearQuantizationConfig, backward_conf: LinearQuantizationConfig
        ) -> None:
            super().__init__()
            self.autograd_func = autograd_func
            self.register_buffer(
                "forw_min_value", forward_conf.min_value
            )
            self.register_buffer(
                "forw_max_value", forward_conf.max_value
            )
            self.register_buffer(
                "backw_min_value", backward_conf.min_value
            )
            self.register_buffer(
                "backw_max_value", backward_conf.max_value
            )
            self._kwargs = {
                "forw_min_value": self.forw_min_value,
                "forw_max_value": self.forw_max_value,
                "backw_min_value": self.backw_min_value,
                "backw_max_value": self.backw_max_value,
            }

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            return self.autograd_func.apply(
                x,
                self.forw_min_value,
                self.forw_max_value,
                self.backw_min_value,
                self.backw_max_value,
            )

    return LinearQuantizationForwBackwQuantizationModule


QuantizeForwHTE: type[_ForwModule] = get_linear_quantization_forw_module(
    QuantizeForwHTEAutograd
)
QuantizeForwStochastic: type[_ForwModule] = get_linear_quantization_forw_module(
    QuantizeForwStochasticAutograd
)
QuantizeForwHTEBackwHTE: type[_ForwBackwModule] = get_linear_quantization_forwbackw_module(
    QuantizeForwHTEBackwHTEAutograd
)
QuantizeForwHTEBackwStochastic: type[_ForwBackwModule] = (
    get_linear_quantization_forwbackw_module(QuantizeForwHTEBackwStochasticAutograd)
)
QuantizeForwStochasticBackwStochastic: type[_ForwBackwModule] = (
    get_linear_quantization_forwbackw_module(QuantizeForwStochasticBackwStochasticAutograd)
)
