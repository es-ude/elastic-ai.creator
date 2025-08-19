from dataclasses import dataclass
from typing import Protocol, TypeVar, Union, cast, overload, runtime_checkable

T = TypeVar("T", bound="ConvertableToFixedPointValues")


@runtime_checkable
class ConvertableToFixedPointValues(Protocol[T]):
    def round(self: T) -> T: ...

    def int(self: T) -> T: ...

    def float(self: T) -> T: ...

    def __gt__(self: T, other: Union[int, float, T]) -> T:  # type: ignore
        ...

    def __lt__(self: T, other: Union[int, float, T]) -> T:  # type: ignore
        ...

    def __or__(self: T, other: T) -> T: ...

    def __mul__(self: T, other: Union[int, T, float]) -> T:  # type: ignore
        ...

    def __truediv__(self: T, other: Union[int, float]) -> T:  # type: ignore
        ...


@dataclass
class FixedPointConfig:
    total_bits: int
    frac_bits: int

    @property
    def minimum_as_integer(self) -> int:
        return 2 ** (self.total_bits - 1) * (-1)

    @property
    def maximum_as_integer(self) -> int:
        return 2 ** (self.total_bits - 1) - 1

    @property
    def minimum_as_rational(self) -> float:
        return -1 * (1 << (self.total_bits - 1)) / (1 << self.frac_bits)

    @property
    def minimum_step_as_rational(self) -> float:
        return 1 / (1 << self.frac_bits)

    @property
    def maximum_as_rational(self) -> float:
        return int("1" * (self.total_bits - 1), 2) / (1 << self.frac_bits)

    @overload
    def integer_out_of_bounds(self, number: T) -> T: ...

    @overload
    def integer_out_of_bounds(self, number: float | int) -> bool: ...

    def integer_out_of_bounds(self, number: float | int | T) -> bool | T:
        return (number < self.minimum_as_integer) | (number > self.maximum_as_integer)

    @overload
    def rational_out_of_bounds(self, number: T) -> T: ...

    @overload
    def rational_out_of_bounds(self, number: float | int) -> bool: ...

    def rational_out_of_bounds(self, number: float | int | T) -> bool | T:
        return (number < self.minimum_as_rational) | (number > self.maximum_as_rational)

    @overload
    def cut_as_integer(self, number: float | int) -> int: ...

    @overload
    def cut_as_integer(self, number: T) -> T: ...

    def cut_as_integer(self, number: float | int | list | T) -> int | list | T:
        """Cutting input number to integer directly (more like in hardware)"""
        if isinstance(number, ConvertableToFixedPointValues):
            return self._cut_T_to_integer(cast(T, number))
        elif isinstance(number, list):
            return list(map(self.cut_as_integer, number))
        return self._cut_float_or_int_to_integer(number)

    def _cut_T_to_integer(self, number: T) -> T:
        return (number * (1 << self.frac_bits)).int().float()

    def _cut_float_or_int_to_integer(self, number: float | int) -> int:
        return int(number * (1 << self.frac_bits))

    @overload
    def round_to_integer(self, number: float | int) -> int: ...

    @overload
    def round_to_integer(self, number: T) -> T: ...

    def round_to_integer(self, number: float | int | T) -> int | T:
        """Mathematical Round function for number"""
        if isinstance(number, ConvertableToFixedPointValues):
            return self._round_T_to_integer(cast(T, number))
        return self._round_float_or_int_to_integer(number)

    def _round_T_to_integer(self, number: T) -> T:
        return (number * (1 << self.frac_bits)).round().int().float()

    def _round_float_or_int_to_integer(self, number: float | int) -> int:
        return round(number * (1 << self.frac_bits))

    @overload
    def as_rational(self, number: float | int) -> float: ...

    @overload
    def as_rational(self, number: T) -> T: ...

    def as_rational(self, number: float | int | T) -> float | T:
        return number / (1 << self.frac_bits)

    def integer_to_binary_string(self, number: int) -> str:
        if self.integer_out_of_bounds(number):
            raise ValueError(
                f"Value '{number}' cannot be represented with {self.total_bits} bits."
            )
        if number < 0:
            twos = (1 << self.total_bits) + number
        else:
            twos = number
        return f"{twos:0{self.total_bits}b}"
