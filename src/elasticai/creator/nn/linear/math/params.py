from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Self, TypeVar, overload, runtime_checkable

from numpy import clip


@runtime_checkable
class ConvertableToIntegerValues[T: ConvertableToIntegerValues](Protocol):
    def round(self) -> Self: ...

    @overload
    def clamp(self, min: Self, max: Self) -> Self: ...

    @overload
    def clamp(
        self,
        min: bool | complex | float | int | None = None,
        max: bool | complex | float | int | None = None,
    ) -> Self: ...

    def is_floating_point(self: Self) -> bool: ...

    def __gt__(self, other: int | float | Self) -> Self: ...

    def __lt__(self, other: int | float | Self) -> Self: ...

    def __or__(self, other: Self) -> Self: ...

    def __mul__(sel, other: int | Self | float) -> Self: ...

    def __truediv__(self, other: int | float) -> Self: ...
    def __add__(self, other: Self | int | float) -> Self: ...
    def __sub__(self, other: Self | int | float) -> Self: ...
    @overload
    def __and__(self, other: Self) -> Self: ...
    @overload
    def __and__(self, other: Self | bool) -> Self: ...
    @overload
    def __and__(self, other: bool | complex | float | int) -> Self: ...

    @overload
    def not_equal(self, other: Self) -> Self: ...
    @overload
    def not_equal(self, other: int | float) -> Self: ...

    def int(self) -> Self: ...

    def float(self) -> Self: ...

    def abs(self) -> Self: ...


_T = TypeVar("_T", bound=ConvertableToIntegerValues)


@dataclass(frozen=True)
class LinearParams(ABC):
    """Paramter for linear quantization.

    Attributes:
        float_range(tuple[float,float]): range of floating point values
        total_bits(int): total with of integer representation
        singed(bool): weither the integer values are signed (Default: True)

    Raises:
        ValueError: if values are invalid
    """

    float_range: tuple[float, float]
    total_bits: int
    signed: bool = True

    def __post_init__(self):
        if self.float_range[0] >= self.float_range[1]:
            raise ValueError("min_float has to be smaller than max_float!")
        if self.total_bits <= 0:
            raise ValueError("total_bits needs to be positivie!")

    @property
    @abstractmethod
    def scale_factor(self) -> float: ...

    @property
    def zero_point(self) -> int:
        return clip(
            a=round(self.float_range[0] / self.scale_factor) + self.minimum_as_integer,
            a_min=self.minimum_as_integer,
            a_max=self.maximum_as_integer,
        )

    @property
    @abstractmethod
    def minimum_as_integer(self) -> int: ...

    @property
    @abstractmethod
    def maximum_as_integer(self) -> int: ...

    @property
    def minimum_as_rational(self) -> float:
        return self.scale_factor * (self.minimum_as_integer - self.zero_point)

    @property
    def maximum_as_rational(self) -> float:
        return self.scale_factor * (self.maximum_as_integer - self.zero_point)

    @overload
    def integer_out_overflow(self, number: _T) -> _T: ...

    @overload
    def integer_out_overflow(self, number: int) -> bool: ...

    def integer_out_overflow(self, number: int | _T) -> bool | _T:
        return number > self.maximum_as_integer

    @overload
    def integer_out_underflow(self, number: _T) -> _T: ...

    @overload
    def integer_out_underflow(self, number: int) -> bool: ...

    def integer_out_underflow(self, number: int | _T) -> bool | _T:
        return number < self.minimum_as_integer

    @overload
    def integer_out_of_bounds(self, number: _T) -> _T: ...

    @overload
    def integer_out_of_bounds(self, number: int) -> bool: ...

    def integer_out_of_bounds(self, number: int | _T) -> bool | _T:
        if isinstance(number, ConvertableToIntegerValues):
            return self._check_integer_out_of_bounds(number)
        else:
            return self._check_integer_out_of_bounds(number)

    def _check_integer_out_of_bounds(self, number) -> bool | _T:
        return self.integer_out_underflow(number) | self.integer_out_overflow(number)

    @overload
    def rational_out_overflow(self, number: _T) -> _T: ...

    @overload
    def rational_out_overflow(self, number: float) -> bool: ...

    def rational_out_overflow(self, number: float | _T) -> bool | _T:
        return number > self.maximum_as_rational

    @overload
    def rational_out_underflow(self, number: _T) -> _T: ...

    @overload
    def rational_out_underflow(self, number: float) -> bool: ...

    def rational_out_underflow(self, number: float | _T) -> bool | _T:
        return number < self.minimum_as_rational

    @overload
    def rational_out_of_bounds(self, number: _T) -> _T: ...

    @overload
    def rational_out_of_bounds(self, number: float) -> bool: ...

    def rational_out_of_bounds(self, number) -> bool | _T:
        return self.rational_out_underflow(number) | self.rational_out_overflow(number)


class AsymetricLinearParams(LinearParams):
    """Paramter for asymetric linear quantization.

    Attributes:
        float_range(tuple[float,float]): range of floating point values
        total_bits(int): total with of integer representation
        singed(bool): weither the integer values are signed (Default: True)

    Raises:
        ValueError: if values are invalid
    """

    def __post_init__(self):
        if self.float_range[0] >= self.float_range[1]:
            raise ValueError("min_float has to be smaller than max_float!")
        if self.total_bits <= 0:
            raise ValueError("total_bits needs to be positivie!")

    @property
    def scale_factor(self) -> float:
        return (self.float_range[1] - self.float_range[0]) / (2**self.total_bits - 1)

    @property
    def minimum_as_integer(self) -> int:
        return 0 if not self.signed else (-1) * 2 ** (self.total_bits - 1)

    @property
    def maximum_as_integer(self) -> int:
        return (
            2**self.total_bits - 1
            if not self.signed
            else 2 ** (self.total_bits - 1) - 1
        )


class SymetricLinearParams(LinearParams):
    """Paramter for symmetric linear quantization.

    Attributes:
        float_range(tuple[float,float]): range of floating point values
        total_bits(int): total with of integer representation
        singed(bool): weither the integer values are signed (Default: True)

    Raises:
        ValueError: if values are invalid
    """

    def __post_init__(self):
        if self.float_range[0] >= self.float_range[1]:
            raise ValueError("min_float has to be smaller than max_float!")
        if self.total_bits <= 0:
            raise ValueError("total_bits needs to be positivie!")

    @property
    def scale_factor(self) -> float:
        return (self.float_range[1] - self.float_range[0]) / (2**self.total_bits - 2)

    @property
    def minimum_as_integer(self) -> int:
        return 0 if not self.signed else (-1) * (2 ** (self.total_bits - 1) - 1)

    @property
    def maximum_as_integer(self) -> int:
        return (
            2**self.total_bits - 1
            if not self.signed
            else 2 ** (self.total_bits - 1) - 1
        )


if __name__ == "__main__":
    asym_params = AsymetricLinearParams(
        float_range=(-1.0, 1.0), total_bits=8, signed=True
    )
    print("Scale Factor:", asym_params.scale_factor)
    print("Zero Point:", asym_params.zero_point)
    print("Minimum as Integer:", asym_params.minimum_as_integer)
    print("Maximum as Integer:", asym_params.maximum_as_integer)
    print("Minimum as Rational:", asym_params.minimum_as_rational)
    print("Maximum as Rational:", asym_params.maximum_as_rational)

    sym_params = SymetricLinearParams(
        float_range=(-1.0, 1.0), total_bits=8, signed=True
    )
    print("Scale Factor:", sym_params.scale_factor)
    print("Zero Point:", sym_params.zero_point)
    print("Minimum as Integer:", sym_params.minimum_as_integer)
    print("Maximum as Integer:", sym_params.maximum_as_integer)
    print("Minimum as Rational:", sym_params.minimum_as_rational)
    print("Maximum as Rational:", sym_params.maximum_as_rational)
