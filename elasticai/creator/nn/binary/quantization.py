from torch import Tensor

from .math_operations import MathOperations as _BinaryOperations


def quantize(x: Tensor) -> Tensor:
    return _BinaryOperations().quantize(x)
