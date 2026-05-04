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
    def minimum_as_integer(self) -> int:
        return self._config.minimum_as_integer

    @property
    def maximum_as_integer(self) -> int:
        return self._config.maximum_as_integer

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
        """Cutting the input number to integer directly (more like in hardware)"""
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

    @overload
    def clamp(self, number: int) -> int: ...

    @overload
    def clamp(self, number: float) -> float: ...

    @overload
    def clamp(self, number: T) -> T: ...

    def clamp(self, number: float | int | T) -> float | int | T:
        if isinstance(number, ConvertableToFixedPointValues):
            return self._clamp_T_to_T(cast(T, number))
        elif isinstance(number, float):
            return self._clamp_for_float(number)
        else:
            return self._clamp_for_integer(number)

    def _clamp_T_to_T(self, number: T) -> T:
        return number.clamp(min=self.minimum_as_rational, max=self.maximum_as_rational)

    def _clamp_for_float(self, number: float) -> float:
        if self._config.rational_out_overflow(number):
            return self._config.maximum_as_rational
        elif self._config.rational_out_underflow(number):
            return self._config.minimum_as_rational
        else:
            return number

    def _clamp_for_integer(self, number: int) -> int:
        if self._config.integer_out_overflow(number):
            return self._config.maximum_as_integer
        elif self._config.integer_out_underflow(number):
            return self._config.minimum_as_integer
        else:
            return number

    @overload
    def to_twos(self, number: int) -> int: ...

    @overload
    def to_twos(self, number: float) -> int: ...

    @overload
    def to_twos(self, number: T) -> T: ...

    def to_twos(self, number: int | float | T) -> int | float | T:
        if isinstance(number, ConvertableToFixedPointValues):
            return self._to_twos_for_T(cast(T, number))
        elif isinstance(number, float):
            value = self.round_to_integer(number)
            return (
                self._to_twos_for_integer(value) * self._config.minimum_step_as_rational
            )
        else:
            return self._to_twos_for_integer(number)

    def _to_twos_for_integer(self, number: int) -> int:
        return number & ((1 << self.total_bits) - 1)

    def _to_twos_for_T(self, number: T) -> T:
        if number.is_floating_point():
            value = self.round_to_integer(number).int()
            return (
                value & ((1 << self.total_bits) - 1)
            ) * self._config.minimum_step_as_rational
        else:
            return number & ((1 << self.total_bits) - 1)

    @overload
    def is_power_of_2(self, number: T) -> T: ...

    @overload
    def is_power_of_2(self, number: int) -> bool: ...

    @overload
    def is_power_of_2(self, number: float) -> bool: ...

    def is_power_of_2(self, number: int | float | T) -> bool | T:
        if isinstance(number, ConvertableToFixedPointValues):
            return self._is_power_of_2_from_T(cast(T, number))
        elif isinstance(number, float):
            raise ValueError("Put in the integer value not the float value.")
        else:
            return self._is_power_of_2_from_int(number)

    @staticmethod
    def _is_power_of_2_from_int(number: int) -> bool:
        value = abs(number)
        return value != 0 and (value & (value - 1)) == 0

    @staticmethod
    def _is_power_of_2_from_T(number: T) -> T:
        value = number.abs()
        return (value != 0) & ((value & (value - 1)) == 0)
