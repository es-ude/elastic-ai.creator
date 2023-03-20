from dataclasses import dataclass
from typing import Protocol, TypeVar, Union, cast, overload, runtime_checkable

T = TypeVar("T", bound="ConvertableToFixedPointValues")


@runtime_checkable
class ConvertableToFixedPointValues(Protocol[T]):
    def round(self: T) -> T:
        ...

    def int(self: T) -> T:
        ...

    def float(self: T) -> T:
        ...

    def __gt__(self: T, other: Union[int, float, T]) -> T:  # type: ignore
        ...

    def __lt__(self: T, other: Union[int, float, T]) -> T:  # type: ignore
        ...

    def __or__(self: T, other: T) -> T:
        ...

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
    def maximum_as_rational(self) -> float:
        return int("1" * (self.total_bits - 1), 2) / (1 << self.frac_bits)

    def integer_out_of_bounds(self, number: T) -> T:
        return (number < self.minimum_as_integer) | (number > self.maximum_as_integer)

    def rational_out_of_bounds(self, number: T) -> T:
        return (number < self.minimum_as_rational) | (number > self.maximum_as_rational)

    @overload
    def as_integer(self, number: float | int) -> int:
        ...

    @overload
    def as_integer(self, number: T) -> T:
        ...

    def as_integer(self, number: float | int | T) -> int | T:
        if isinstance(number, ConvertableToFixedPointValues):
            return self._convert_T_to_integer(cast(T, number))
        return self._convert_float_or_int_to_integer(number)

    @overload
    def as_rational(self, number: float | int) -> float:
        ...

    @overload
    def as_rational(self, number: T) -> T:
        ...

    def as_rational(self, number: float | int | T) -> float | T:
        return number / (1 << self.frac_bits)

    def _convert_T_to_integer(self, number: T) -> T:
        return (number * (1 << self.frac_bits)).int().float()

    def _convert_float_or_int_to_integer(self, number: float | int) -> int:
        return round(number * (1 << self.frac_bits))
