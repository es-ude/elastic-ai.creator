import dataclasses

from .number_conversion import (
    bits_to_integer,
    bits_to_rational,
    convert_rational_to_bit_pattern,
    integer_to_bits,
    max_integer,
    max_natural,
    max_rational,
    min_integer,
    min_natural,
    min_rational,
)


@dataclasses.dataclass
class FXPParams:
    total_bits: int
    frac_bits: int


class NumberConverter:
    def __init__(self, fxp_params: FXPParams):
        self._fxp_params = fxp_params

    def bits_to_integer(self, pattern: str) -> int:
        return bits_to_integer(pattern)

    def bits_to_rational(self, pattern: str) -> float:
        return bits_to_rational(pattern, frac_bits=self._fxp_params.frac_bits)

    def rational_to_bits(self, rational: float) -> str:
        return convert_rational_to_bit_pattern(
            rational=rational,
            total_bits=self._fxp_params.total_bits,
            frac_bits=self._fxp_params.frac_bits,
        )

    def bits_to_natural(self, pattern: str) -> int:
        return bits_to_rational(pattern, frac_bits=self._fxp_params.frac_bits)

    def integer_to_bits(self, number: int) -> str:
        return integer_to_bits(number, total_bits=self._fxp_params.total_bits)

    @property
    def max_rational(self) -> float:
        return max_rational(
            total_bits=self._fxp_params.total_bits, frac_bits=self._fxp_params.frac_bits
        )

    @property
    def max_integer(self) -> int:
        return max_integer(total_bits=self._fxp_params.total_bits)

    @property
    def min_rational(self) -> float:
        return min_rational(
            total_bits=self._fxp_params.total_bits, frac_bits=self._fxp_params.frac_bits
        )

    @property
    def min_integer(self) -> int:
        return min_integer(total_bits=self._fxp_params.total_bits)

    @property
    def max_natural(self) -> int:
        return max_natural(total_bits=self._fxp_params.total_bits)

    @property
    def min_natural(self) -> int:
        return min_natural(total_bits=self._fxp_params.total_bits)
