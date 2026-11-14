from typing import Protocol

import torch
from torch import Tensor

from elasticai.creator.base_modules.math_operations import Quantize


class MathOperations(Quantize, Protocol): ...


class HardSigmoid(torch.nn.Hardsigmoid):
    def __init__(self, mathops: MathOperations, slope: float = 1 / 4) -> None:
        super().__init__(inplace=False)
        self._mathops = mathops
        self._slope = slope

    def forward(self, input: Tensor) -> Tensor:
        return self._mathops.quantize(
            torch.clamp(input=self._slope * input + 0.5, min=0.0, max=1.0)
        )
