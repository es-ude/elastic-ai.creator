import warnings
from dataclasses import dataclass
from typing import Any, TypeVar, cast, overload

from ._int_arith_protocol import IntArithmetic
from .fxp_params import (
    ConvertableToFixedPointValues,
    FxpParams,
)

T = TypeVar("T", bound="ConvertableToFixedPointValues")


@dataclass
class FxpArithmetic(IntArithmetic):
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
    def cut_as_integer(self, number: list[float | int]) -> list[float | int]: ...

    @overload
    def cut_as_integer(
        self, number: list[list[float | int]]
    ) -> list[list[float | int]]: ...

    @overload
    def cut_as_integer(self, number: T) -> T: ...

    def cut_as_integer(self, number: float | int | T | list) -> int | list | T:
        """Cut the input number to integer directly (more like in hardware)"""
        match number:
            case list():
                warnings.warn(
                    "Calling cut_to_integer with lists is deprecated and will be removed in the future. Use functions from itertools at call site instead.",
                    stacklevel=2,
                )
                return cast(
                    list,
                    self._cut_nested_list_to_integer(number),
                )
            case float() | int():
                return self._cut_float_or_int_to_integer(number)
            case ConvertableToFixedPointValues():
                return self._cut_T_to_integer(number)

    def _cut_nested_list_to_integer(self, numbers: Any) -> Any:
        if isinstance(numbers, list):
            return list(map(self._cut_nested_list_to_integer, numbers))
        else:
            return self.cut_as_integer(numbers)

    def _cut_T_to_integer(self, number: T) -> T:
        return (number / self._config.minimum_step_as_rational).int().float()

    def _cut_float_or_int_to_integer(self, number: float | int) -> int:
        return int(number / self._config.minimum_step_as_rational)

    @overload
    def cut_as_rational(self, number: float) -> float: ...

    @overload
    def cut_as_rational(self, number: list[float]) -> list[float]: ...

    @overload
    def cut_as_rational(self, number: list[list[float]]) -> list[list[float]]: ...

    @overload
    def cut_as_rational(self, number: T) -> T: ...

    def cut_as_rational(self, number: float | T | list) -> float | list | T:
        """Cut the input number to integer directly (more like in hardware)"""
        match number:
            case list():
                warnings.warn(
                    "Calling cut_to_integer with lists is deprecated and will be removed in the future. Use functions from itertools at call site instead.",
                    stacklevel=2,
                )
                return cast(
                    list,
                    self._cut_nested_list_to_rational(number),
                )
            case float() | int():
                return self._cut_float_or_int_to_rational(number)
            case ConvertableToFixedPointValues():
                return self._cut_T_to_rational(number)

    def _cut_nested_list_to_rational(self, numbers: Any) -> Any:
        if isinstance(numbers, list):
            return list(map(self._cut_nested_list_to_rational, numbers))
        else:
            return self.cut_as_rational(numbers)

    def _cut_T_to_rational(self, number: T) -> T:
        return self._cut_T_to_integer(number) * self._config.minimum_step_as_rational

    def _cut_float_or_int_to_rational(self, number: float) -> float:
        return float(
            self.cut_as_integer(number) * self._config.minimum_step_as_rational
        )

    @overload
    def round_to_integer(self, number: float | int) -> int: ...

    @overload
    def round_to_integer(self, number: T) -> T: ...

    def round_to_integer(self, number: float | int | T) -> int | T:
        """Mathematical Round function for number"""
        match number:
            case ConvertableToFixedPointValues() as n:
                return self._round_T_to_integer(n)
            case int() | float():
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
    def round_to_rational(self, number: float) -> float: ...

    @overload
    def round_to_rational(self, number: T) -> T: ...

    def round_to_rational(self, number: float | T) -> Any:
        """Mathematical Round function for number"""
        match number:
            case list():
                warnings.warn(
                    "Calling cut_to_integer with lists is deprecated and will be removed in the future. Use functions from itertools at call site instead.",
                    stacklevel=2,
                )
                return cast(
                    list,
                    self._round_nested_list_to_rational(number),
                )
            case float():
                return self._round_float_or_int_to_rational(number)
            case ConvertableToFixedPointValues() as n:
                return self._round_T_to_rational(n)

    def _round_nested_list_to_rational(self, numbers: Any) -> Any:
        if isinstance(numbers, list):
            return list(map(self._round_nested_list_to_rational, numbers))
        else:
            return self.round_to_rational(numbers)

    def _round_T_to_rational(self, number: T) -> T:
        return self._round_T_to_integer(number) * self._config.minimum_step_as_rational

    def _round_float_or_int_to_rational(self, number: float) -> float:
        return self.round_to_integer(number) * self._config.minimum_step_as_rational

    @overload
    def clamp(self, number: int) -> int: ...

    @overload
    def clamp(self, number: float) -> float: ...

    @overload
    def clamp(self, number: T) -> T: ...

    def clamp(self, number: float | int | T) -> float | int | T:
        match number:
            case float():
                return self._clamp_float(number)
            case int():
                return self._clamp_int(number)
            case ConvertableToFixedPointValues():
                return self._clamp_T_to_T(number)

    def _clamp_T_to_T(self, number: T) -> T:
        return number.clamp(min=self.minimum_as_rational, max=self.maximum_as_rational)

    def _clamp_float(self, number: float) -> float:
        if self._config.rational_out_overflow(number):
            return self._config.maximum_as_rational
        elif self._config.rational_out_underflow(number):
            return self._config.minimum_as_rational
        else:
            return number

    def _clamp_int(self, number: int) -> int:
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
        match number:
            case ConvertableToFixedPointValues():
                return self._to_twos_for_T(number)
            case float():
                value = self.round_to_integer(number)
                return (
                    self._to_twos_for_integer(value)
                    * self._config.minimum_step_as_rational
                )
            case int():
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

    def is_power_of_2(self, number: int | T) -> bool | T:
        match number:
            case int():
                return self._is_power_of_2_from_int(number)
            case ConvertableToFixedPointValues():
                return self._is_power_of_2_from_T(number)

    @staticmethod
    def _is_power_of_2_from_int(number: int) -> bool:
        value = abs(number)
        return value != 0 and (value & (value - 1)) == 0

    @staticmethod
    def _is_power_of_2_from_T(number: T) -> T:
        value = number.abs()
        return (value.not_equal(0)) & ((value & (value - 1)) == 0)
