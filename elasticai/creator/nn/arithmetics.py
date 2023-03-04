from typing import Protocol

import torch

from elasticai.creator.nn.autograd_functions.fixed_point_quantization import (
    FixedPointDequantFunction,
    FixedPointQuantFunction,
)
from elasticai.creator.two_complement_fixed_point_config import (
    TwoComplementFixedPointConfig,
)


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
    def __init__(self, config: TwoComplementFixedPointConfig) -> None:
        self.config = config

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return self.round(self.clamp(a))

    def clamp(self, a: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            a, min=self.config.minimum_as_rational, max=self.config.maximum_as_rational
        )

    def round(self, a: torch.Tensor) -> torch.Tensor:
        def float_to_int(x: torch.Tensor) -> torch.Tensor:
            return FixedPointQuantFunction.apply(x, self.config)

        def int_to_fixed_point(x: torch.Tensor) -> torch.Tensor:
            return FixedPointDequantFunction.apply(x, self.config)

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
