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
from .two_complement_fixed_point_config import FixedPointConfigV2


class _ForwModule(Module):
    def __init__(self, forward_conf: FixedPointConfigV2): ...


class _ForwBackwModule(Module):
    def __init__(
        self, forward_conf: FixedPointConfigV2, backward_conf: FixedPointConfigV2
    ): ...


def get_fxp_forw_quantization_module(
    autograd_func: torch.autograd.Function,
) -> type[_ForwModule]:
    class FxPForwQuantizationModule(Module):
        def __init__(self, forward_conf: FixedPointConfigV2) -> None:
            super().__init__()
            self.autograd_func = autograd_func
            self.register_buffer(
                "forw_resolution_per_int", forward_conf.resolution_per_int
            )
            self.register_buffer(
                "forw_minimum_as_rational", forward_conf.minimum_as_rational
            )
            self.register_buffer(
                "forw_maximum_as_rational", forward_conf.maximum_as_rational
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.autograd_func.apply(
                x,
                self.forw_resolution_per_int,
                self.forw_minimum_as_rational,
                self.forw_maximum_as_rational,
            )

    return FxPForwQuantizationModule


def get_fxp_forwbackw_quantization_module(
    autograd_func: torch.autograd.Function,
) -> type[_ForwBackwModule]:
    class FxPForwBackwQuantizationModule(Module):
        def __init__(
            self, forward_conf: FixedPointConfigV2, backward_conf: FixedPointConfigV2
        ) -> None:
            super().__init__()
            self.autograd_func = autograd_func
            self.register_buffer(
                "forw_resolution_per_int", forward_conf.resolution_per_int
            )
            self.register_buffer(
                "forw_minimum_as_rational", forward_conf.minimum_as_rational
            )
            self.register_buffer(
                "forw_maximum_as_rational", forward_conf.maximum_as_rational
            )
            self.register_buffer(
                "backw_resolution_per_int", backward_conf.resolution_per_int
            )
            self.register_buffer(
                "backw_minimum_as_rational", backward_conf.minimum_as_rational
            )
            self.register_buffer(
                "backw_maximum_as_rational", backward_conf.maximum_as_rational
            )
            self._kwargs = {
                "forw_resolution_per_int": self.forw_resolution_per_int,
                "forw_minimum_as_rational": self.forw_minimum_as_rational,
                "forw_maximum_as_rational": self.forw_maximum_as_rational,
                "backw_resolution_per_int": self.backw_resolution_per_int,
                "backw_minimum_as_rational": self.backw_minimum_as_rational,
                "backw_maximum_as_rational": self.backw_maximum_as_rational,
            }

        def forward(self, x: Tensor) -> Tensor:
            return self.autograd_func.apply(
                x,
                self.forw_resolution_per_int,
                self.forw_minimum_as_rational,
                self.forw_maximum_as_rational,
                self.backw_resolution_per_int,
                self.backw_minimum_as_rational,
                self.backw_maximum_as_rational,
            )

    return FxPForwBackwQuantizationModule


QuantizeForwHTE: type[_ForwModule] = get_fxp_forw_quantization_module(
    QuantizeForwHTEAutograd
)
QuantizeForwStochastic: type[_ForwModule] = get_fxp_forw_quantization_module(
    QuantizeForwStochasticAutograd
)
QuantizeForwHTEBackwHTE: type[_ForwBackwModule] = get_fxp_forwbackw_quantization_module(
    QuantizeForwHTEBackwHTEAutograd
)
QuantizeForwHTEBackwStochastic: type[_ForwBackwModule] = (
    get_fxp_forwbackw_quantization_module(QuantizeForwHTEBackwStochasticAutograd)
)
QuantizeForwStochasticBackwStochastic: type[_ForwBackwModule] = (
    get_fxp_forwbackw_quantization_module(QuantizeForwStochasticBackwStochasticAutograd)
)
