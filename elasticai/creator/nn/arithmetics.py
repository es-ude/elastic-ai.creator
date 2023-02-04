from typing import Protocol

import torch

from elasticai.creator.nn.autograd_functions.fixed_point_quantization import (
    FixedPointDequantFunction,
    FixedPointQuantFunction,
)
from elasticai.creator.vhdl.number_representations import FixedPointFactory


class Arithmetics(Protocol):
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        ...

    def clamp(self, a: torch.Tensor) -> torch.Tensor:
        ...

    def round(self, a: torch.Tensor) -> torch.Tensor:
        ...

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ...

    def sum(self, tensor: torch.Tensor, *tensors: torch.Tensor) -> torch.Tensor:
        ...

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ...

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ...


class FloatArithmetics(Arithmetics):
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def clamp(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def round(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def sum(self, tensor: torch.Tensor, *tensors: torch.Tensor) -> torch.Tensor:
        summed = tensor
        for t in tensors:
            summed += t
        return summed

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class FixedPointArithmetics(Arithmetics):
    def __init__(self, fixed_point_factory: FixedPointFactory) -> None:
        self.fixed_point_factory = fixed_point_factory

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return self.round(self.clamp(a))

    def clamp(self, a: torch.Tensor) -> torch.Tensor:
        total_bits, frac_bits = (
            self.fixed_point_factory.total_bits,
            self.fixed_point_factory.frac_bits,
        )
        min_fp = -1 * (1 << (total_bits - 1)) / (1 << frac_bits)
        max_fp = int("1" * (total_bits - 1), 2) / (1 << frac_bits)
        return torch.clamp(a, min=min_fp, max=max_fp)

    def round(self, a: torch.Tensor) -> torch.Tensor:
        def float_to_int(x: torch.Tensor) -> torch.Tensor:
            return FixedPointQuantFunction.apply(x, self.fixed_point_factory)

        def int_to_fixed_point(x: torch.Tensor) -> torch.Tensor:
            return FixedPointDequantFunction.apply(x, self.fixed_point_factory)

        return int_to_fixed_point(float_to_int(a))

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.clamp(a + b)

    def sum(self, tensor: torch.Tensor, *tensors: torch.Tensor) -> torch.Tensor:
        summed = tensor
        for t in tensors:
            summed += t
        return self.clamp(summed)

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.round(self.clamp(a * b))

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.round(self.clamp(torch.matmul(a, b)))
