from dataclasses import dataclass
from typing import Protocol, TypeVar, Union, cast, overload, runtime_checkable

T = TypeVar("T", bound="ConvertableToIntegerValues")


@runtime_checkable
class ConvertableToIntegerValues(Protocol[T]):
    def round(self: T) -> T: ...

    def int(self: T) -> T: ...

    def float(self: T) -> T: ...

    def clamp(self: T, min: Union[int, T] = None, max: Union[int, T] = None) -> T: ...

    def __gt__(self: T, other: Union[int, T]) -> T:  # type: ignore
        ...

    def __lt__(self: T, other: Union[int, T]) -> T:  # type: ignore
        ...

    def __or__(self: T, other: T) -> T: ...

    def __mul__(self: T, other: Union[int, T]) -> T:  # type: ignore
        ...

    def __truediv__(self: T, other: Union[int]) -> T:  # type: ignore
        ...


@dataclass(frozen=True)
class IntParams:
    total_bits: int
    signed: bool = True

    def __post_init__(self):
        if self.total_bits <= 0:
            raise Exception(
                f"total bits need to be > 0 for {self.__class__.__name__}. "
                f"You have set {self.total_bits=}."
            )

    @property
    def minimum_value(self) -> int:
        return 2 ** (self.total_bits - 1) * (-1) if self.signed else 0

    @property
    def maximum_value(self) -> int:
        return 2 ** (self.total_bits - 1) - 1 if self.signed else 2**self.total_bits - 1

    @overload
    def integer_out_overflow(self, number: T) -> T: ...

    @overload
    def integer_out_overflow(self, number: int) -> bool: ...

    def integer_out_overflow(self, number: int | T) -> bool | T:
        return number > self.maximum_value

    @overload
    def integer_out_underflow(self, number: T) -> T: ...

    @overload
    def integer_out_underflow(self, number: int) -> bool: ...

    def integer_out_underflow(self, number: int | T) -> bool | T:
        return number < self.minimum_value

    @overload
    def integer_out_of_bounds(self, number: T) -> T: ...

    @overload
    def integer_out_of_bounds(self, number: int) -> bool: ...

    def integer_out_of_bounds(self, number: int | T) -> bool | T:
        if isinstance(number, ConvertableToIntegerValues):
            return self._check_integer_out_of_bounds(cast(T, number))
        else:
            return self._check_integer_out_of_bounds(number)

    def _check_integer_out_of_bounds(self, number) -> bool | T:
        return self.integer_out_underflow(number) | self.integer_out_overflow(number)

    @overload
    def is_power_of_2(self, number: T) -> T: ...

    @overload
    def is_power_of_2(self, number: int) -> bool: ...

    def is_power_of_2(self, number: int | T) -> bool | T:
        if isinstance(number, ConvertableToIntegerValues):
            return self._is_power_of_2_from_T(cast(T, number))
        else:
            return self._is_power_of_2_from_int(number)

    @staticmethod
    def _is_power_of_2_from_int(number: int) -> bool:
        return number != 0 and (number & (number - 1)) == 0

    @staticmethod
    def _is_power_of_2_from_T(number: T) -> T:
        return (number != 0) & ((number & (number - 1)) == 0)
