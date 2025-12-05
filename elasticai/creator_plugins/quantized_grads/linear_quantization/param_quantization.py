from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module

from . import quantize_linear_asym_hte, quantize_linear_asym_stochastic
from .autograd import QuantizeSimulatedLinearAsymForwHTEAutograd, QuantizeSimulatedLinearAsymForwStochasticAutograd, \
    QuantizeSimulatedIntForwHTEAutograd, QuantizeSimulatedIntForwStochasticAutograd
from .linear_quantization_config import LinearAsymQuantizationConfig, IntQuantizationConfig


class ParamQuantizationSimulatedModule(Module):
    def __init__(self, config: LinearAsymQuantizationConfig):
        super().__init__()
        self.register_buffer("min_value", config.min_value)
        self.register_buffer("max_value", config.max_value)
        self.config = config
        self.autograd_quantize_simulated: torch.autograd.Function

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def right_inverse(self, x: Tensor) -> Tensor:
        raise NotImplementedError

class ParamQuantizationModule(ParamQuantizationSimulatedModule):
    def __init__(self, config: LinearAsymQuantizationConfig):
        super().__init__(config)
        self.quantize_func: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]

    def quantize(self, tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.quantize_func(tensor, self.min_value, self.max_value)

def get_quantize_to_int_simulated_quantization(
        autograd_quantize_simulated: torch.autograd.Function) -> type[ParamQuantizationSimulatedModule]:
    class QuantizeSimulatedSTE(ParamQuantizationSimulatedModule):
        def __init__(self, config: IntQuantizationConfig):
            super().__init__(config)
            self.autograd_quantize_simulated = autograd_quantize_simulated

        def forward(self, x: Tensor) -> Tensor:
            return self.autograd_quantize_simulated.apply(
                x,
                self.min_value,
                self.max_value,
            )

        def right_inverse(self, x: Tensor) -> Tensor:
            return x
            return self.autograd_quantize_simulated.apply(
                x,
                self.min_value,
                self.max_value,
            )

    return QuantizeSimulatedSTE


def get_quantize_to_linear_quantization(
        autograd_quantize_simulated: torch.autograd.Function,
        quantize_func: Callable[[Tensor,Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
) -> type[ParamQuantizationModule]:
    class QuantizeToQuantizationSTE(ParamQuantizationModule):
        def __init__(self, config: LinearAsymQuantizationConfig):
            super().__init__(config)
            self.autograd_quantize_simulated = autograd_quantize_simulated
            self.quantize_func = quantize_func

        def forward(self, x: Tensor) -> Tensor:
            return self.autograd_quantize_simulated.apply(
                x,
                self.min_value,
                self.max_value,
            )

        def right_inverse(self, x: Tensor) -> Tensor:
            return x

    return QuantizeToQuantizationSTE

QuantizeParamSTEToLinearAsymQuantizationHTE = get_quantize_to_linear_quantization(QuantizeSimulatedLinearAsymForwHTEAutograd, quantize_linear_asym_hte)
QuantizeParamSTEToLinearAsymQuantizationStochastic = get_quantize_to_linear_quantization(QuantizeSimulatedLinearAsymForwStochasticAutograd, quantize_linear_asym_stochastic)

QuantizeParamSTEToIntHTE = get_quantize_to_int_simulated_quantization(QuantizeSimulatedIntForwHTEAutograd)
QuantizeParamSTEToIntStochastic = get_quantize_to_int_simulated_quantization(QuantizeSimulatedIntForwStochasticAutograd)

class QuantizeTensorToLinearAsymQuantizationHTE(QuantizeParamSTEToLinearAsymQuantizationHTE):
    """
    This Modules can be used for Tensor quantization
    """

class QuantizeTensorToLinearAsymQuantizationStochastic(QuantizeParamSTEToLinearAsymQuantizationStochastic):
    """
    This Modules can be used for Tensor quantization
    """

class QuantizeTensorToIntHTE(QuantizeParamSTEToIntHTE):
    """
    This Modules can be used for Tensor quantization
    """

class QuantizeTensorToIntStochastic(QuantizeParamSTEToIntStochastic):
    """
    This Modules can be used for Tensor quantization
    """
