from typing import cast

import torch

from elasticai.creator.base_modules.math_operations import Add, MatMul, Mul, Quantize

from ._round_to_float import RoundToFloat


class MathOperations(Quantize, Add, MatMul, Mul):
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
        return self._round(self._clamp(a))

    def _clamp(self, a: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            a, min=self.smallest_negative_value, max=self.largest_positive_value
        )

    def _round(self, a: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            RoundToFloat.apply(a, self.mantissa_bits, self.exponent_bits),
        )

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a + b)

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(a * b)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.quantize(torch.matmul(a, b))
