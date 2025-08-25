from dataclasses import dataclass
from typing import Protocol, TypeVar, Union, cast, overload, runtime_checkable

import torch

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
class FxpParams:
    """
    This class behaves almost like a frozen instance,
    but calculates and sets minimum_as_rational_tensor and maximum_as_rational_tensor just once in __post__init__.
    After the __post_init__ the __setattr__ and __getattr__ methods block the change of variables.
    """

    total_bits: int
    frac_bits: int
    signed: bool = True

    def __post_init__(self):
        """
        This python API call for dataclass is executed after init.
        This way we check if the FxpParams is valid while keeping the config immutable.
        """
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
    def minimum_as_rational_tensor(self):
        return torch.Tensor([self.minimum_as_rational])

    @property
    def minimum_step_as_rational(self) -> float:
        return 1 / (1 << self.frac_bits)

    @property
    def resolution_per_int(self):
        return torch.Tensor([2**self.frac_bits])

    @property
    def maximum_as_rational(self) -> float:
        return self.maximum_as_integer * self.minimum_step_as_rational

    @property
    def maximum_as_rational_tensor(self):
        return torch.Tensor([self.maximum_as_rational])

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
            return self._chck_integer_out_of_bounds(cast(T, number))
        else:
            return self._chck_integer_out_of_bounds(number)

    def _chck_integer_out_of_bounds(self, number: int | T) -> bool | T:
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

    def rational_out_of_bounds(self, number: float | T) -> bool | T:
        return self.rational_out_underflow(number) | self.rational_out_overflow(number)
