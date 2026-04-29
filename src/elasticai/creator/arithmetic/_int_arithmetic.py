from ._int_arith_protocol import IntArithmetic
from .fxp_arithmetic import FxpArithmetic
from .fxp_params import FxpParams


def int_arithmetic(total_bits: int, signed: bool) -> IntArithmetic:
    return FxpArithmetic(FxpParams(total_bits=total_bits, frac_bits=0, signed=signed))
