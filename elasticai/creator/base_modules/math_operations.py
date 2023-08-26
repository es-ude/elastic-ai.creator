from typing import Protocol

from torch import Tensor


class Quantize(Protocol):
    def quantize(self, a: Tensor) -> Tensor:
        ...


class MatMul(Protocol):
    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        ...


class Add(Protocol):
    def add(self, a: Tensor, b: Tensor) -> Tensor:
        ...


class Mul(Protocol):
    def mul(self, a: Tensor, b: Tensor) -> Tensor:
        ...
