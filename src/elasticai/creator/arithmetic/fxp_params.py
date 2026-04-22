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
    def integer_out_overflow(self, number: T) -> T: ...

    @overload
    def integer_out_overflow(self, number: int) -> bool: ...

    def integer_out_overflow(self, number: int | T) -> bool | T:
        return number > self.maximum_as_integer

    @overload
    def integer_out_underflow(self, number: T) -> T: ...

    @overload
    def integer_out_underflow(self, number: int) -> bool: ...

    def integer_out_underflow(self, number: int | T) -> bool | T:
        return number < self.minimum_as_integer

    @overload
    def integer_out_of_bounds(self, number: T) -> T: ...

    @overload
    def integer_out_of_bounds(self, number: int) -> bool: ...

    def integer_out_of_bounds(self, number: int | T) -> bool | T:
        if isinstance(number, ConvertableToFixedPointValues):
            return self._check_integer_out_of_bounds(cast(T, number))
        else:
            return self._check_integer_out_of_bounds(number)

    def _check_integer_out_of_bounds(self, number) -> bool | T:
        return self.integer_out_underflow(number) | self.integer_out_overflow(number)

    @overload
    def rational_out_overflow(self, number: T) -> T: ...

    @overload
    def rational_out_overflow(self, number: float) -> bool: ...

    def rational_out_overflow(self, number: float | T) -> bool | T:
        return number > self.maximum_as_rational

    @overload
    def rational_out_underflow(self, number: T) -> T: ...

    @overload
    def rational_out_underflow(self, number: float) -> bool: ...

    def rational_out_underflow(self, number: float | T) -> bool | T:
        return number < self.minimum_as_rational

    @overload
    def rational_out_of_bounds(self, number: T) -> T: ...

    @overload
    def rational_out_of_bounds(self, number: float) -> bool: ...

    def rational_out_of_bounds(self, number) -> bool | T:
        return self.rational_out_underflow(number) | self.rational_out_overflow(number)
