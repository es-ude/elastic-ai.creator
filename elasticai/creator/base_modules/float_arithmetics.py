from typing import Optional

import torch

from elasticai.creator.base_modules.arithmetics import Arithmetics


class FloatArithmetics(Arithmetics):
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def clamp(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def round(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def sum(
        self, a: torch.Tensor, dim: Optional[int | tuple[int, ...]] = None
    ) -> torch.Tensor:
        return torch.sum(a, dim=dim)

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)
