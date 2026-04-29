from dataclasses import dataclass
from typing import Protocol, Self, TypeVar, overload, runtime_checkable


@runtime_checkable
class ConvertableToFixedPointValues[T: ConvertableToFixedPointValues](Protocol):
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


_T = TypeVar("_T", bound=ConvertableToFixedPointValues)


@dataclass(frozen=True)
class FxpParams:
    total_bits: int
    frac_bits: int
    signed: bool = True

    def __post_init__(self):
        if self.total_bits <= 0:
            raise Exception(
                f"total bits need to be > 0 for {self.__class__.__name__}. "
                f"You have set {self.total_bits=}."
            )
        if self.frac_bits > self.total_bits:
            raise Exception(
                f"total bits-1 needs to be > frac bits for {self.__class__.__name__}. "
                f"You have set {self.total_bits=} and {self.frac_bits=}."
            )

    @property
    def minimum_as_integer(self) -> int:
        return 2 ** (self.total_bits - 1) * (-1) if self.signed else 0

    @property
    def maximum_as_integer(self) -> int:
        return 2 ** (self.total_bits - 1) - 1 if self.signed else 2**self.total_bits - 1

    @property
    def minimum_as_rational(self) -> float:
        return self.minimum_as_integer * self.minimum_step_as_rational

    @property
    def minimum_step_as_rational(self) -> float:
        return 1 / (1 << self.frac_bits)

    @property
    def maximum_as_rational(self) -> float:
        return self.maximum_as_integer * self.minimum_step_as_rational

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
        if isinstance(number, ConvertableToFixedPointValues):
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
