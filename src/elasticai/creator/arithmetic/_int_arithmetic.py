from typing import Protocol, overload

from .fxp_arithmetic import FxpArithmetic
from .fxp_params import FxpParams, T


def int_arithmetic(total_bits: int, signed: bool) -> "IntArithmetic":
    return FxpArithmetic(FxpParams(total_bits=total_bits, frac_bits=0, signed=signed))


class IntArithmetic(Protocol):
    @property
    def config(self) -> FxpParams: ...

    @property
    def total_bits(self) -> int: ...

    def integer_out_of_bounds(self, number): ...

    @property
    def minimum_as_integer(self) -> int: ...

    @property
    def maximum_as_integer(self) -> int: ...

    @overload
    def cut_as_integer(self, number: float | int) -> int:
        """Cutting the input number to integer directly (more like in hardware)"""
        ...

    @overload
    def cut_as_integer(self, number: list) -> list:
        """Cutting the input number to integer directly (more like in hardware)"""
        ...

    @overload
    def cut_as_integer(self, number: T) -> T:
        """Cutting the input number to integer directly (more like in hardware)"""
        ...

    @overload
    def round_to_integer(self, number: float | int) -> int:
        """Mathematical Round function for number"""
        ...

    @overload
    def round_to_integer(self, number: T) -> T:
        """Mathematical Round function for number"""
        ...

    @overload
    def clamp(self, number: int) -> int: ...

    @overload
    def clamp(self, number: float) -> float: ...

    @overload
    def clamp(self, number: T) -> T: ...

    @overload
    def to_twos(self, number: int) -> int: ...

    @overload
    def to_twos(self, number: float) -> int: ...

    @overload
    def to_twos(self, number: T) -> T: ...

    def to_twos(self, number: int | float | T) -> int | T: ...

    @overload
    def is_power_of_2(self, number: T) -> T: ...

    @overload
    def is_power_of_2(self, number: int) -> bool: ...

    @overload
    def is_power_of_2(self, number: float) -> bool: ...

    def is_power_of_2(self, number: int | float | T) -> bool | T: ...
