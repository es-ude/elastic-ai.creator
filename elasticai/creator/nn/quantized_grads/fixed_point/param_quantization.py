from typing import Callable

from torch import Tensor
from torch.nn import Module

from .quantize_to_fixed_point import quantize_to_fxp_hte, quantize_to_fxp_stochastic
from .two_complement_fixed_point_config import FixedPointConfigV2


def get_quantize_to_fixed_point(
    func: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
) -> tuple[type[Module], type[Module]]:
    class QuantizeToFixedPoint(Module):
        def __init__(self, config: FixedPointConfigV2):
            super().__init__()
            self.register_buffer("minimum_as_rational", config.minimum_as_rational)
            self.register_buffer("maximum_as_rational", config.maximum_as_rational)
            self.register_buffer("resolution_per_int", config.resolution_per_int)

        @staticmethod
        def forward(x: Tensor) -> Tensor:
            return x

        def right_inverse(self, x: Tensor) -> Tensor:
            return func(
                x,
                self.resolution_per_int,
                self.minimum_as_rational,
                self.maximum_as_rational,
            )

    class QuantizeToFixedPointSTE(Module):
        def __init__(self, config: FixedPointConfigV2):
            super().__init__()
            self.register_buffer("minimum_as_rational", config.minimum_as_rational)
            self.register_buffer("maximum_as_rational", config.maximum_as_rational)
            self.register_buffer("resolution_per_int", config.resolution_per_int)

        def forward(self, x: Tensor) -> Tensor:
            return func(
                x,
                self.resolution_per_int,
                self.minimum_as_rational,
                self.maximum_as_rational,
            )

        def right_inverse(self, x: Tensor) -> Tensor:
            return x

    return QuantizeToFixedPoint, QuantizeToFixedPointSTE


(QuantizeParamToFixedPointHTE, QuantizeParamSTEToFixedPointHTE) = (
    get_quantize_to_fixed_point(quantize_to_fxp_hte)
)
(QuantizeParamToFixedPointStochastic, QuantizeParamSTEToFixedPointStochastic) = (
    get_quantize_to_fixed_point(quantize_to_fxp_stochastic)
)


class QuantizeTensorToFixedPointHTE(QuantizeParamSTEToFixedPointHTE):
    """
    This Modules can be used for Tensor quantization
    """


class QuantizeTensorToFixedPointStochastic(QuantizeParamSTEToFixedPointStochastic):
    """
    This Modules can be used for Tensor quantization
    """
