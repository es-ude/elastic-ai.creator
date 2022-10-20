import math
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from itertools import chain
from typing import Any, Callable

FixedPointFactory = Callable[[float], "FixedPoint"]


def _assert_range(value: float, total_bits: int, frac_bits: int) -> None:
    max_value = 2 ** (total_bits - frac_bits - 1)
    min_value = max_value * (-1)
    if not min_value <= value < max_value:
        raise ValueError(
            (
                f"Value {value} cannot represented as a fixed point value with {total_bits} total bits "
                f"and {frac_bits} fraction bits (value range: [{min_value}, {max_value}))."
            )
        )


def _assert_is_compatible(fp1: "FixedPoint", fp2: "FixedPoint") -> None:
    if not (fp1.total_bits == fp2.total_bits and fp1.frac_bits == fp2.frac_bits):
        raise ValueError(
            (
                f"FixedPoint objects not compatible (total_bits: {fp1.total_bits} != {fp2.total_bits}); "
                f"frac_bits: {fp1.frac_bits} != {fp2.frac_bits})."
            )
        )


def _invert_int(value: int, num_bits: int) -> int:
    return value ^ int("1" * num_bits, 2)


def _discard_leading_bits(value: int, num_bits: int) -> int:
    return value & int("1" * num_bits, 2)


def _calculate_two_complement(value: int, num_bits: int) -> int:
    return _invert_int(abs(value), num_bits) + 1


class FixedPoint:
    """
    A data type that converts a given number to the corresponding fixed-point representation.
    A fixed-point value is an unsigned integer in two's complement.
    Parameters:
        value (float): Value to be represented as fixed-point value.
        total_bits (int): Total number of bits of the fixed-point representation (including number of fractional bits).
        frac_bits (int): Number of bits to represent the fractional part of the number.
    Examples:
        >>> fixed_point_value = FixedPoint(-2.9, total_bits=8, frac_bits=4)
        >>> int(fixed_point_value)
        210
        >>> float(fixed_point_value)
        -2.875
        >>> fixed_point_value.to_signed_int()
        -46
        >>> fixed_point_value.to_bin()
        '11010010'
        >>> fixed_point_value.to_hex()
        'd2'
    """

    __slots__ = ["_value", "_frac_bits", "_total_bits"]

    def __init__(self, value: float, total_bits: int, frac_bits: int) -> None:
        self._value = float(value)
        self._total_bits = total_bits
        self._frac_bits = frac_bits
        _assert_range(self._value, self._total_bits, self._frac_bits)

    def __int__(self) -> int:
        fp_int = int(self._value * (1 << self._frac_bits))
        if fp_int < 0:
            fp_int = _calculate_two_complement(fp_int, self._total_bits)
        return fp_int

    def __float__(self) -> float:
        return FixedPoint.from_unsigned_int(
            int(self), self._total_bits, self._frac_bits
        )._value

    def __eq__(self, other: Any) -> bool:
        return float(self) == float(other)

    def __ne__(self, other: Any) -> bool:
        return float(self) != float(other)

    def __lt__(self, other: Any) -> bool:
        return float(self) < float(other)

    def __le__(self, other: Any) -> bool:
        return float(self) <= float(other)

    def __gt__(self, other: Any) -> bool:
        return float(self) > float(other)

    def __ge__(self, other: Any) -> bool:
        return float(self) >= float(other)

    def __add__(self, other: "FixedPoint") -> "FixedPoint":
        _assert_is_compatible(self, other)
        return self._identical_fixed_point_from_int(
            _discard_leading_bits(int(self) + int(other), num_bits=self._total_bits)
        )

    def __sub__(self, other: "FixedPoint") -> "FixedPoint":
        return self + (-other)

    def __and__(self, other: "FixedPoint") -> "FixedPoint":
        _assert_is_compatible(self, other)
        return self._identical_fixed_point_from_int(int(self) & int(other))

    def __or__(self, other: "FixedPoint") -> "FixedPoint":
        _assert_is_compatible(self, other)
        return self._identical_fixed_point_from_int(int(self) | int(other))

    def __xor__(self, other: "FixedPoint") -> "FixedPoint":
        _assert_is_compatible(self, other)
        return self._identical_fixed_point_from_int(int(self) ^ int(other))

    def __invert__(self) -> "FixedPoint":
        return self._identical_fixed_point_from_int(
            _invert_int(int(self), num_bits=self._total_bits)
        )

    def __neg__(self) -> "FixedPoint":
        return self._identical_fixed_point(-self._value)

    def __abs__(self) -> "FixedPoint":
        return self._identical_fixed_point(abs(self._value))

    def __str__(self) -> str:
        return str(int(self))

    def __repr__(self) -> str:
        return f"FixedPoint(value={self._value}, total_bits={self._total_bits}, frac_bits={self._frac_bits})"

    def _identical_fixed_point(self, value: float) -> "FixedPoint":
        return FixedPoint(
            value=value, total_bits=self._total_bits, frac_bits=self._frac_bits
        )

    def _identical_fixed_point_from_int(self, value: int) -> "FixedPoint":
        return FixedPoint.from_unsigned_int(
            value=value, total_bits=self._total_bits, frac_bits=self._frac_bits
        )

    @staticmethod
    def from_unsigned_int(value: int, total_bits: int, frac_bits: int) -> "FixedPoint":
        if value > 2**total_bits - 1:
            raise ValueError(
                f"Value {value} cannot interpreted as a fixed point with {total_bits} total bits."
            )
        is_negative = value & (1 << total_bits - 1) > 0
        if is_negative:
            value = _calculate_two_complement(value, total_bits)
            value *= -1
        float_value = value / (1 << frac_bits)
        return FixedPoint(float_value, total_bits=total_bits, frac_bits=frac_bits)

    @staticmethod
    def from_signed_int(value: int, total_bits: int, frac_bits: int) -> "FixedPoint":
        float_value = value / (1 << frac_bits)
        return FixedPoint(float_value, total_bits=total_bits, frac_bits=frac_bits)

    @staticmethod
    def get_factory(total_bits: int, frac_bits: int) -> FixedPointFactory:
        return partial(FixedPoint, total_bits=total_bits, frac_bits=frac_bits)

    @property
    def total_bits(self) -> int:
        return self._total_bits

    @property
    def frac_bits(self) -> int:
        return self._frac_bits

    def bin_iter(self) -> Iterator[int]:
        return ((int(self) >> i) & 1 for i in range(self._total_bits))

    def to_signed_int(self) -> int:
        return int(abs(self)) * (-1 if self < 0 else 1)

    def to_bin(self) -> str:
        return f"{int(self):0{self._total_bits}b}"

    def to_hex(self) -> str:
        return f"{int(self):0{math.ceil(self._total_bits / 4)}x}"


class ClippedFixedPoint(FixedPoint):
    def __init__(self, value: float, total_bits: int, frac_bits: int) -> None:
        max_value = (2 ** (total_bits - 1) - 1) / (1 << frac_bits)
        min_value = 2 ** (total_bits - frac_bits - 1) * (-1)
        if min_value <= value <= max_value:
            super().__init__(value=value, total_bits=total_bits, frac_bits=frac_bits)
        else:
            super().__init__(
                value=max_value if value > max_value else min_value,
                total_bits=total_bits,
                frac_bits=frac_bits,
            )

    def __float__(self) -> float:
        return ClippedFixedPoint.from_unsigned_int(
            int(self), self._total_bits, self._frac_bits
        )._value

    def __repr__(self) -> str:
        return f"ClippedFixedPoint(value={self._value}, total_bits={self._total_bits}, frac_bits={self._frac_bits})"

    def _identical_fixed_point(self, value: float) -> "ClippedFixedPoint":
        return ClippedFixedPoint(
            value=value, total_bits=self._total_bits, frac_bits=self._frac_bits
        )

    def _identical_fixed_point_from_int(self, value: int) -> "ClippedFixedPoint":
        return ClippedFixedPoint.from_unsigned_int(
            value=value, total_bits=self._total_bits, frac_bits=self._frac_bits
        )

    @staticmethod
    def from_unsigned_int(
        value: int, total_bits: int, frac_bits: int
    ) -> "ClippedFixedPoint":
        value = _discard_leading_bits(value, num_bits=total_bits)
        is_negative = value & (1 << total_bits - 1) > 0
        if is_negative:
            value = _calculate_two_complement(value, total_bits)
            value *= -1
        float_value = value / (1 << frac_bits)
        return ClippedFixedPoint(
            float_value, total_bits=total_bits, frac_bits=frac_bits
        )

    @staticmethod
    def from_signed_int(
        value: int, total_bits: int, frac_bits: int
    ) -> "ClippedFixedPoint":
        float_value = value / (1 << frac_bits)
        return ClippedFixedPoint(
            float_value, total_bits=total_bits, frac_bits=frac_bits
        )

    @staticmethod
    def get_factory(
        total_bits: int, frac_bits: int
    ) -> Callable[[float], "ClippedFixedPoint"]:
        return partial(ClippedFixedPoint, total_bits=total_bits, frac_bits=frac_bits)


def infer_total_and_frac_bits(*values: Sequence[FixedPoint]) -> tuple[int, int]:
    if sum(len(value_list) == 0 for value_list in values) > 0:
        raise ValueError("Cannot infer total bits and frac bits from an empty list.")
    total_bits, frac_bits = values[0][0].total_bits, values[0][0].frac_bits
    for value in chain(*values):
        if value.total_bits != total_bits or value.frac_bits != frac_bits:
            raise ValueError(
                "Cannot infer total bits and frac bits from a list with mixed total bits or frac bits."
            )
    return total_bits, frac_bits


def fixed_point_params_from_factory(factory: FixedPointFactory) -> tuple[int, int]:
    return factory(1).total_bits, factory(1).frac_bits


def float_values_to_fixed_point(
    values: list[float], total_bits: int, frac_bits: int
) -> list[FixedPoint]:
    return list(map(lambda x: FixedPoint(x, total_bits, frac_bits), values))


def unsigned_int_values_to_fixed_point(
    values: list[int], total_bits: int, frac_bits: int
) -> list[FixedPoint]:
    return list(
        map(lambda x: FixedPoint.from_unsigned_int(x, total_bits, frac_bits), values)
    )


class ToLogicEncoder:
    """
    Throughout our implementations we have to deal with two different levels of representations for numbers:
    During training we typically need to apply mathematical operations and we do not care too much about how our numbers are encoded.
    E.g. in a scenario where we want to use two bit on hardware to represent our numbers, in our machine learning framework we
    might decide it is beneficial to use the numbers -3 and 4 for some reason. However, especially in the context of precomputed
    results, the hardware implementation does not need to know the numeric values, but instead just needs to be able to keep a
    consistent and correct mapping. The NumericToLogicEncoder takes care of performing the translations from numeric representation
    to the bit vector used in the hardware implementation. We encode bit vectors just as unsigned integers.
    """

    def __init__(self):
        self._symbols = set()
        self._mapping = dict()

    def _update_mapping(self) -> None:
        sorted_numerics = sorted(self._symbols)
        mapping = {value: index for index, value in enumerate(sorted_numerics)}
        self._mapping.update(mapping)

    def register_symbol(self, numeric_representation: int) -> None:
        self._symbols.add(numeric_representation)
        self._update_mapping()

    def register_symbols(self, symbols: Iterable[int]) -> None:
        for symbol in symbols:
            self._symbols.add(symbol)
        self._update_mapping()

    @staticmethod
    def _int_to_bin(number: int, num_bits: int) -> str:
        return f"{number:0{num_bits}b}"

    @property
    def bit_width(self) -> int:
        return math.floor(math.log2(len(self._symbols)))

    def __len__(self):
        return len(self._symbols)

    def __eq__(self, other: "ToLogicEncoder") -> bool:
        return self._symbols == other._symbols and self._mapping == other._mapping

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for symbol, encoded_symbol in self._mapping.values():
            yield symbol, encoded_symbol

    def __getitem__(self, item: int) -> int:
        return self._mapping[item]

    def __call__(self, number: int) -> str:
        if number not in self._symbols:
            raise ValueError
        return self._int_to_bin(self._mapping[number], self.bit_width)
