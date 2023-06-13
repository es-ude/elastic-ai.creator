from typing import Optional, cast

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

    def sum(
        self, a: torch.Tensor, dim: Optional[int | tuple[int, ...]] = None
    ) -> torch.Tensor:
        return self.quantize(torch.sum(a, dim=dim))

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a * b)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.matmul(a, b))

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
        outputs = torch.nn.functional.conv1d(
            input=inputs,
            weight=weights,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        return self.quantize(outputs)
