from typing import Protocol

import torch
from torch import Tensor, nn

from elasticai.creator.base_modules.math_operations import (
    ParamExponent,
    Quantize,
    TwosScale,
)


class MathOperations(Quantize, TwosScale, ParamExponent, Protocol): ...


class PReLU2(torch.nn.PReLU):
    def __init__(
        self,
        math_operations: MathOperations,
        num_parameters: int = 1,
        init: float = 0.25,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_parameters=num_parameters, init=init, device=device, dtype=dtype
        )
        self._mathops = math_operations

    def _get_weight_exponent(self) -> Tensor:
        return self._mathops.get_exponent(self.weight)

    def _quantize_weights_to_twos(self) -> Tensor:
        e_weight = self._get_weight_exponent()
        weight = self._mathops.twos_scaling(e_weight)
        return self._mathops.quantize(weight)

    def forward(self, input: Tensor) -> Tensor:
        weight = self._quantize_weights_to_twos()
        return self._mathops.quantize(nn.functional.prelu(input, weight))
