from torch import Tensor

from ._math_operations import MathOperations as _FxpOperations
from ._two_complement_fixed_point_config import FixedPointConfig as _FxpConfig


def quantize(x: Tensor, total_bits: int, frac_bits: int) -> Tensor:
    return _FxpOperations(
        config=_FxpConfig(total_bits=total_bits, frac_bits=frac_bits)
    ).quantize(x)
