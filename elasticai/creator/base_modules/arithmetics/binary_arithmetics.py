from typing import cast

import torch

from elasticai.creator.base_modules.autograd_functions.binary_quantization import (
    Binarize,
)

from .arithmetics import Arithmetics


class BinaryArithmetics(Arithmetics):
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, Binarize.apply(a))

    def clamp(self, a: torch.Tensor) -> torch.Tensor:
        return self.quantize(a)

    def round(self, a: torch.Tensor) -> torch.Tensor:
        return self.quantize(a)

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a + b)

    def sum(self, a: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.sum(a, *others))

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a * b)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.matmul(a, b))
