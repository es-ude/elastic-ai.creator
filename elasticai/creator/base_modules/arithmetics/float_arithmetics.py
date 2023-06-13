from typing import Optional

import torch

from .arithmetics import Arithmetics


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

    def conv1d(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor | None,
        stride: int,
        padding: int | str,
        dilation: int,
        groups: int,
    ) -> torch.Tensor:
        return torch.nn.functional.conv1d(
            input=inputs,
            weight=weights,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
