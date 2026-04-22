from dataclasses import dataclass
from typing import cast, overload

from elasticai.creator.arithmetic.fxp_params import (
    ConvertableToFixedPointValues,
    FxpParams,
    T,
)


@dataclass
class FxpArithmetic:
    def __init__(self, fxp_params: FxpParams):
        self._config = fxp_params

    @property
    def config(self) -> FxpParams:
        return self._config

    @property
    def total_bits(self) -> int:
        return self._config.total_bits

    @property
    def frac_bits(self) -> int:
        return self._config.frac_bits

    @property
    def minimum_as_rational(self) -> float:
        return self._config.minimum_as_rational

    @property
    def maximum_as_rational(self) -> float:
        return self._config.maximum_as_rational

    def integer_out_of_bounds(self, number):
        return self._config.integer_out_of_bounds(number)

    @overload
    def cut_as_integer(self, number: float | int) -> int: ...

    @overload
    def cut_as_integer(self, number: list) -> list: ...

    @overload
    def cut_as_integer(self, number: T) -> T: ...

    def cut_as_integer(self, number: float | int | list | T) -> int | list | T:
        """Cutting input number to integer directly (more like in hardware)"""
        if isinstance(number, ConvertableToFixedPointValues):
            return self._cut_T_to_integer(cast(T, number))
        elif isinstance(number, list):
            return list(map(self.cut_as_integer, number))
        else:
            return self._cut_float_or_int_to_integer(number)

    def _cut_T_to_integer(self, number: T) -> T:
        return (number / self._config.minimum_step_as_rational).int().float()

    def _cut_float_or_int_to_integer(self, number: float | int) -> int:
        return int(number / self._config.minimum_step_as_rational)

    @overload
    def round_to_integer(self, number: float | int) -> int: ...

    @overload
    def round_to_integer(self, number: T) -> T: ...

    def round_to_integer(self, number: float | int | T) -> int | T:
        """Mathematical Round function for number"""
        if isinstance(number, ConvertableToFixedPointValues):
            return self._round_T_to_integer(cast(T, number))
        else:
            return self._round_float_or_int_to_integer(number)

    def _round_T_to_integer(self, number: T) -> T:
        return (number / self._config.minimum_step_as_rational).round().int().float()

    def _round_float_or_int_to_integer(self, number: float | int) -> int:
        return round(number / self._config.minimum_step_as_rational)

    @overload
    def as_rational(self, number: int) -> float: ...

    @overload
    def as_rational(self, number: T) -> T: ...

    def as_rational(self, number: int | T) -> float | T:
        return number * self._config.minimum_step_as_rational
