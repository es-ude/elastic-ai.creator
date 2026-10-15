from typing import Protocol

import torch
from torch import Tensor, nn

from elasticai.creator.base_modules.math_operations import (
    ParamExponent,
    Quantize,
    TwosScale,
)


class MathOperations(Quantize, TwosScale, ParamExponent, Protocol): ...


class LeakyReLU2(torch.nn.LeakyReLU):
    def __init__(
        self,
        math_operations: MathOperations,
        init: float = 0.125,
    ) -> None:
        self._mathops = math_operations
        self.exp = self._mathops.get_exponent(Tensor([init]))
        self.slope = float(2 ** int(self.exp))
        super().__init__(negative_slope=self.slope)

    def _get_weight_exponent(self) -> Tensor:
        return self.exp

    def _quantize_weights_to_twos(self) -> Tensor:
        e_weight = self._get_weight_exponent()
        weight = self._mathops.twos_scaling(e_weight)
        return self._mathops.quantize(weight)

    def forward(self, input: Tensor) -> Tensor:
        return self._mathops.quantize(nn.functional.leaky_relu(input, self.slope))
