from typing import Optional

import torch

from elasticai.creator.base_modules.arithmetics import Arithmetics
from elasticai.creator.base_modules.autograd_functions.binary_quantization import (
    Binarize,
)


class BinaryArithmetics(Arithmetics):
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return Binarize.apply(a)

    def clamp(self, a: torch.Tensor) -> torch.Tensor:
        return self.quantize(a)

    def round(self, a: torch.Tensor) -> torch.Tensor:
        return self.quantize(a)

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a + b)

    def sum(
        self, a: torch.Tensor, dim: Optional[int | tuple[int, ...]] = None
    ) -> torch.Tensor:
        return self.quantize(torch.sum(a, dim=dim))

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a * b)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.matmul(a, b))
