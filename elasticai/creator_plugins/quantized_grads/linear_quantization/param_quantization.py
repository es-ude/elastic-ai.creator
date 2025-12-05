from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module

from . import quantize_linear_asym_hte, quantize_linear_asym_stochastic
from .autograd import QuantizeFakeLinearAsymForwHTEAutograd, QuantizeFakeLinearAsymForwStochasticAutograd, \
    QuantizeFakeIntForwHTEAutograd, QuantizeFakeIntForwStochasticAutograd
from .linear_quantization_config import LinearQuantizationConfig

class ParamQuantizationModule(Module):
    def __init__(self, config: LinearQuantizationConfig):
        super().__init__()
        self.register_buffer("min_value", config.min_value)
        self.register_buffer("max_value", config.max_value)
        self.config = config
        self.autograd_quantize_fake: torch.autograd.Function
        self.quantize_func: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]

    def quantize(self, tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.quantize_func(tensor, self.min_value, self.max_value)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def right_inverse(self, x: Tensor) -> Tensor:
        raise NotImplementedError


def get_quantize_to_linear_quantization(
        autograd_quantize_fake: torch.autograd.Function,
        quantize_func: Callable[[Tensor,Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]=None,
) -> tuple[type[Module], type[Module]]:
    class QuantizeToQuantizationSTE(ParamQuantizationModule):
        def __init__(self, config: LinearQuantizationConfig):
            super().__init__(config)
            self.autograd_quantize_fake = autograd_quantize_fake
            self.quantize_func = quantize_func

        def forward(self, x: Tensor) -> Tensor:
            return self.autograd_quantize_fake.apply(
                x,
                self.min_value,
                self.max_value,
            )

        def right_inverse(self, x: Tensor) -> Tensor:
            return self.autograd_quantize_fake.apply(
                x,
                self.min_value,
                self.max_value,
            )

    return QuantizeToQuantizationSTE


QuantizeParamSTEToLinearQuantizationHTE = get_quantize_to_linear_quantization(QuantizeFakeLinearAsymForwHTEAutograd, quantize_linear_asym_hte)
QuantizeParamSTEToLinearQuantizationStochastic = get_quantize_to_linear_quantization(QuantizeFakeLinearAsymForwStochasticAutograd, quantize_linear_asym_stochastic)

QuantizeParamSTEToIntHTE = get_quantize_to_linear_quantization(QuantizeFakeIntForwHTEAutograd)
QuantizeParamSTEToIntStochastic = get_quantize_to_linear_quantization(QuantizeFakeIntForwStochasticAutograd)

class QuantizeTensorToLinearQuantizationHTE(QuantizeParamSTEToLinearQuantizationHTE):
    """
    This Modules can be used for Tensor quantization
    """

class QuantizeTensorToLinearQuantizationStochastic(QuantizeParamSTEToLinearQuantizationStochastic):
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
