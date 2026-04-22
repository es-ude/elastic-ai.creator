from typing import Any, Protocol

import torch

from elasticai.creator.base_modules.math_operations import Add, MatMul, Quantize


class MathOperations(Quantize, Add, MatMul, Protocol): ...


class Linear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        operations: MathOperations,
        bias: bool,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self._operations = operations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._operations.quantize(self.weight)
        if self.bias is None:
            return self._operations.matmul(x, weight.T)
        else:
            return self._operations.add(
                self._operations.matmul(x, weight.T),
                self._operations.quantize(self.bias),
            )
