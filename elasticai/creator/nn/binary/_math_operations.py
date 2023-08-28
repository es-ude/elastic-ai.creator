from typing import cast

import torch

from elasticai.creator.base_modules.math_operations import Add, MatMul, Mul, Quantize

from ._binary_quantization_function import Binarize


class MathOperations(Quantize, Add, MatMul, Mul):
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, Binarize.apply(a))

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a + b)

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a * b)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.matmul(a, b))
