from abc import abstractmethod
from typing import Protocol

from torch import Tensor


class Quantize(Protocol):
    @abstractmethod
    def quantize(self, x: Tensor) -> Tensor:
        ...


class MatMul(Protocol):
    @abstractmethod
    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        ...


class Add(Protocol):
    @abstractmethod
    def add(self, a: Tensor, b: Tensor) -> Tensor:
        ...
