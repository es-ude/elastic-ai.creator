from typing import Protocol

import torch
from torch import Tensor, nn

from elasticai.creator.base_modules.math_operations import Quantize


class MathOperations(Quantize, Protocol): ...


class PReLU(torch.nn.PReLU):
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

    def forward(self, input: Tensor) -> Tensor:
        weight = self._mathops.quantize(self.weight)
        return self._mathops.quantize(nn.functional.prelu(input, weight))
