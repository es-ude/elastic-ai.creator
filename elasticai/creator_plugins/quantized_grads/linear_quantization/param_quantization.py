from typing import Callable

from torch import Tensor
from torch.nn import Module

from .quantize_linear import quantize_linear_hte, quantize_linear_stochastic, quantize_linear_stochastic_fake, \
    quantize_linear_hte_fake
from .linear_quantization_config import LinearQuantizationConfig
from .quantize_to_int_with_linear_quantization_style import quantize_to_int_hte_fake, quantize_to_int_stochastic_fake, quantize_to_int_hte, \
    quantize_to_int_stochastic

class ParamLinearQuantizationModule(Module):
    def __init__(self, config: LinearQuantizationConfig):
        super().__init__()
        self.register_buffer("min_value", config.min_value)
        self.register_buffer("max_value", config.max_value)
        self.config = config
        self.quantize: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def right_inverse(self, x: Tensor) -> Tensor:
        raise NotImplementedError


def get_quantize_to_linear_quantization(
        quantize_fake: Callable[[Tensor, Tensor, Tensor], Tensor],
        quantize: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
) -> tuple[type[Module], type[Module]]:
    class QuantizeToLinearQuantization(ParamLinearQuantizationModule):
        def __init__(self, config: LinearQuantizationConfig):
            super().__init__(config)
            self.quantize = quantize

        def forward(self, x: Tensor) -> Tensor:
            return x

        def right_inverse(self, x: Tensor) -> Tensor:
            return quantize_fake(
                x,
                self.min_value,
                self.max_value,
            )

    class QuantizeToLinearQuantizationSTE(ParamLinearQuantizationModule):
        def __init__(self, config: LinearQuantizationConfig):
            super().__init__(config)
            self.quantize = quantize

        def forward(self, x: Tensor) -> Tensor:
            return quantize_fake(
                x,
                self.min_value,
                self.max_value,
            )

        def right_inverse(self, x: Tensor) -> Tensor:
            return x

    return QuantizeToLinearQuantization, QuantizeToLinearQuantizationSTE


(QuantizeParamToLinearQuantizationHTE, QuantizeParamSTEToLinearQuantizationHTE) = (
    get_quantize_to_linear_quantization(quantize_linear_hte_fake, quantize_linear_hte)
)
(QuantizeParamToLinearQuantizationStochastic, QuantizeParamSTEToLinearQuantizationStochastic) = (
    get_quantize_to_linear_quantization(quantize_linear_stochastic_fake, quantize_linear_stochastic)
)

(QuantizeParamToIntHTE, QuantizeParamSTEToIntHTE) = (
    get_quantize_to_linear_quantization(quantize_to_int_hte_fake, quantize_to_int_hte)
)
(QuantizeParamToIntStochastic, QuantizeParamSTEToIntStochastic) = (
    get_quantize_to_linear_quantization(quantize_to_int_stochastic_fake, quantize_to_int_stochastic)
)



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