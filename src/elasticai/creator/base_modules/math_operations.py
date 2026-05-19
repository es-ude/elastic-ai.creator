from abc import abstractmethod
from typing import Protocol

from torch import Tensor


class Quantize(Protocol):
    @abstractmethod
    def quantize(self, a: Tensor) -> Tensor: ...


class MatMul(Protocol):
    @abstractmethod
    def matmul(self, a: Tensor, b: Tensor) -> Tensor: ...


class Add(Protocol):
    @abstractmethod
    def add(self, a: Tensor, b: Tensor) -> Tensor: ...


class Mul(Protocol):
    @abstractmethod
    def mul(self, a: Tensor, b: Tensor) -> Tensor: ...


class TwosScale(Protocol):
    @abstractmethod
    def twos_scaling(self, a: Tensor) -> Tensor: ...


class ParamExponent(Protocol):
    @abstractmethod
    def get_exponent(self, a: Tensor) -> Tensor: ...
