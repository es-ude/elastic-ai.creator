from typing import Any, Protocol

import torch

from elasticai.creator.base_modules.math_operations import Add, MatMul, Quantize


class MathOperations(Quantize, Add, MatMul, Protocol):
    ...


class Linear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        arithmetics: MathOperations,
        bias: bool,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.ops = arithmetics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.ops.quantize(self.weight)

        if self.bias is not None:
            bias = self.ops.quantize(self.bias)
            return self.ops.add(self.ops.matmul(x, weight.T), bias)

        return self.ops.matmul(x, weight.T)
