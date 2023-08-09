from typing import Optional, cast

import torch

from elasticai.creator.base_modules.arithmetics.arithmetics import Arithmetics
from elasticai.creator.base_modules.autograd_functions.round_to_float import (
    RoundToFloat,
)


class FloatArithmetics(Arithmetics):
    def __init__(self, mantissa_bits: int, exponent_bits: int) -> None:
        self.mantissa_bits = mantissa_bits
        self.exponent_bits = exponent_bits

    @property
    def largest_positive_value(self) -> float:
        exponent_bias = 2 ** (self.exponent_bits - 1)
        return (2 - 1 / 2**self.mantissa_bits) * 2 ** (
            2**self.exponent_bits - exponent_bias - 1
        )

    @property
    def smallest_negative_value(self) -> float:
        return -self.largest_positive_value

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return self.round(self.clamp(a))

    def clamp(self, a: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            a, min=self.smallest_negative_value, max=self.largest_positive_value
        )

    def round(self, a: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            RoundToFloat.apply(a, self.mantissa_bits, self.exponent_bits),
        )

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
        bias: Optional[torch.Tensor],
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
