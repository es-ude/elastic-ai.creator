from typing import Protocol

import torch
from torch import Tensor, nn

from elasticai.creator.base_modules.math_operations import Quantize


class MathOperations(Quantize, Protocol): ...


class LeakyReLU(torch.nn.LeakyReLU):
    def __init__(self, mathops: MathOperations, init: float = 1e-2) -> None:
        super().__init__(negative_slope=float(mathops.config.round_to_rational(init)))  # type: ignore
        self._mathops = mathops

    def forward(self, input: Tensor) -> Tensor:
        return self._mathops.quantize(
            nn.functional.leaky_relu(input, self.negative_slope)
        )
