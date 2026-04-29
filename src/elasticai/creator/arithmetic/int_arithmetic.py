from dataclasses import dataclass
from typing import cast, overload

from elasticai.creator.arithmetic.int_params import (
    ConvertableToIntegerValues,
    IntParams,
    T,
)


@dataclass
class IntArithmetic:
    def __init__(self, fxp_params: IntParams):
        self._config: IntParams = fxp_params

    @property
    def config(self) -> IntParams:
        return self._config

    @property
    def total_bits(self) -> int:
        return self._config.total_bits

    @property
    def minimum_value(self) -> int:
        return self._config.minimum_value

    @property
    def maximum_value(self) -> int:
        return self._config.maximum_value

    def integer_out_of_bounds(self, number):
        return self._config.integer_out_of_bounds(number)

    def is_power_of_2(self, number: int) -> bool:
        return self._config.is_power_of_2(number)

    @overload
    def cut_as_integer(self, number: float | int) -> int: ...

    @overload
    def cut_as_integer(self, number: list) -> list: ...

    @overload
    def cut_as_integer(self, number: T) -> T: ...

    def cut_as_integer(self, number: float | int | list | T) -> int | list | T:
        """Cutting input number to integer directly (more like in hardware)"""
        if isinstance(number, ConvertableToIntegerValues):
            return self._cut_T_to_integer(cast(T, number))
        elif isinstance(number, list):
            return list(map(self.cut_as_integer, number))
        else:
            return self._cut_float_or_int_to_integer(number)

    def _cut_T_to_integer(self, number: T) -> T:
        return number.int().float()

    def _cut_float_or_int_to_integer(self, number: float | int) -> int:
        return int(number)

    @overload
    def round_to_integer(self, number: float | int) -> int: ...

    @overload
    def round_to_integer(self, number: T) -> T: ...

    def round_to_integer(self, number: float | int | T) -> int | T:
        """Mathematical Round function for number"""
        if isinstance(number, ConvertableToIntegerValues):
            return self._round_T_to_integer(cast(T, number))
        else:
            return self._round_float_or_int_to_integer(number)

    def _round_T_to_integer(self, number: T) -> T:
        return number.round().int().float()

    def _round_float_or_int_to_integer(self, number: float | int) -> int:
        return round(number)

    @overload
    def clamp(self, number: int) -> int: ...

    @overload
    def clamp(self, number: T) -> T: ...

    def clamp(self, number: int | T) -> int | T:
        if isinstance(number, ConvertableToIntegerValues):
            return self._clamp_T_to_T(cast(T, number))
        else:
            return self._clamp_for_integer(number)

    def _clamp_T_to_T(self, number: T) -> T:
        return number.clamp(min=self.minimum_value, max=self.maximum_value)

    def _clamp_for_integer(self, number: int) -> int:
        if self._config.integer_out_overflow(number):
            return self._config.maximum_value
        elif self._config.integer_out_underflow(number):
            return self._config.minimum_value
        else:
            return number

    @overload
    def to_twos(self, number: int) -> int: ...

    @overload
    def to_twos(self, number: T) -> T: ...

    def to_twos(self, number: int | T) -> int | T:
        if isinstance(number, ConvertableToIntegerValues):
            return self._to_twos_for_integer(cast(T, number))
        else:
            return self._to_twos_for_integer(number)

    def _to_twos_for_integer(self, number: int) -> int:
        return number & ((1 << self.total_bits) - 1)
