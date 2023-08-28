from torch import Tensor

from ._math_operations import MathOperations as _FlpOperations


def quantize(x: Tensor, mantissa_bits: int, exponent_bits: int) -> Tensor:
    return _FlpOperations(
        mantissa_bits=mantissa_bits, exponent_bits=exponent_bits
    ).quantize(x)
