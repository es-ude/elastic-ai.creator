import math
from typing import Any, Iterable, Iterator, Union


class FixedPoint:
    __slots__ = ["_value", "_frac_bits", "_total_bits"]

    def __init__(
        self,
        value: float,
        total_bits: int,
        frac_bits: int,
    ) -> None:
        self._value = float(value)
        self._total_bits = total_bits
        self._frac_bits = frac_bits
        FixedPoint._assert_range(self._value, self._total_bits, self._frac_bits)

    def __int__(self) -> int:
        fp_int = int(self._value * (1 << self._frac_bits))
        if fp_int < 0:
            fp_int = FixedPoint._calculate_two_complement(fp_int, self._total_bits)
        return fp_int

    def __float__(self) -> float:
        return FixedPoint._calculate_float_from_fixed_point(
            int(self), self._total_bits, self._frac_bits
        )

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
        FixedPoint._assert_is_compatible(self, other)
        return self._identical_fixed_point_from_int(
            FixedPoint.discard_leading_bits(
                int(self) + int(other), num_bits=self._total_bits
            )
        )

    def __sub__(self, other: "FixedPoint") -> "FixedPoint":
        return self + (-other)

    def __and__(self, other: "FixedPoint") -> "FixedPoint":
        FixedPoint._assert_is_compatible(self, other)
        return self._identical_fixed_point_from_int(int(self) & int(other))

    def __or__(self, other: "FixedPoint") -> "FixedPoint":
        FixedPoint._assert_is_compatible(self, other)
        return self._identical_fixed_point_from_int(int(self) | int(other))

    def __xor__(self, other: "FixedPoint") -> "FixedPoint":
        FixedPoint._assert_is_compatible(self, other)
        return self._identical_fixed_point_from_int(int(self) ^ int(other))

    def __invert__(self) -> "FixedPoint":
        return self._identical_fixed_point_from_int(
            FixedPoint._invert_int(int(self), num_bits=self._total_bits)
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
        return FixedPoint.from_int(
            value=value, total_bits=self._total_bits, frac_bits=self._frac_bits
        )

    @staticmethod
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

    @staticmethod
    def _assert_is_compatible(fp1: "FixedPoint", fp2: "FixedPoint") -> None:
        if not (fp1.total_bits == fp2.total_bits and fp1.frac_bits == fp2.frac_bits):
            raise ValueError(
                (
                    f"FixedPoint objects not compatible (total_bits: {fp1.total_bits} != {fp2.total_bits}); "
                    f"frac_bits: {fp1.frac_bits} != {fp2.frac_bits})."
                )
            )

    @staticmethod
    def _invert_int(value: int, num_bits: int) -> int:
        return value ^ int("1" * num_bits, 2)

    @staticmethod
    def discard_leading_bits(value: int, num_bits: int) -> int:
        return value & int("1" * num_bits, 2)

    @staticmethod
    def _calculate_two_complement(value: int, num_bits: int) -> int:
        return FixedPoint._invert_int(abs(value), num_bits) + 1

    @staticmethod
    def _calculate_float_from_fixed_point(
        value: int, total_bits: int, frac_bits: int
    ) -> float:
        if value > 2**total_bits - 1:
            raise ValueError(
                f"Value {value} cannot interpreted as a fixed point with {total_bits} total bits."
            )
        is_negative = value & (1 << total_bits - 1) > 0
        if is_negative:
            value = FixedPoint._calculate_two_complement(value, total_bits)
            value *= -1
        return value / (1 << frac_bits)

    @staticmethod
    def from_int(
        value: int,
        total_bits: int,
        frac_bits: int,
    ) -> "FixedPoint":
        return FixedPoint(
            value=FixedPoint._calculate_float_from_fixed_point(
                value, total_bits=total_bits, frac_bits=frac_bits
            ),
            total_bits=total_bits,
            frac_bits=frac_bits,
        )

    @property
    def total_bits(self) -> int:
        return self._total_bits

    @property
    def frac_bits(self) -> int:
        return self._frac_bits

    def bin_iter(self) -> Iterator[int]:
        return ((int(self) >> i) & 1 for i in range(self._total_bits))

    def to_bin(self) -> str:
        return f'"{int(self):0{self._total_bits}b}"'

    def to_hex(self) -> str:
        return f'x"{int(self):0{math.ceil(self._total_bits / 4)}x}"'


class FloatToSignedFixedPointConverter:
    """
    Create a fixed point representation as an unsigned int data type using two complements.

    We might want to have this create its own type `FixedPointNumber` in
    the future. That way we could make sure that the conversion is idempotent
    for numbers that are fixed point already.
    """

    def __init__(self, bits_used_for_fraction: int, strict=True):
        self.bits_used_for_fraction = bits_used_for_fraction
        self._strict = strict

    @property
    def one(self) -> int:
        return 1 << self.bits_used_for_fraction

    def __call__(self, x: float) -> int:
        x_tmp = float(x)
        x_tmp = x_tmp * self.one
        if self._strict and not x_tmp.is_integer():
            raise ValueError(
                f"{x} not convertible to fixed point number using {self.bits_used_for_fraction} bits for fractional part"
            )
        return int(x_tmp)

    def to_string(self, x: float) -> str:
        return str(self.__call__(x))


class FloatToBinaryFixedPointStringConverter:
    def __init__(
        self,
        total_bit_width: int,
        as_signed_fixed_point: FloatToSignedFixedPointConverter,
    ):
        self.total_bit_width = total_bit_width
        self.as_signed_fixed_point = as_signed_fixed_point

    def __call__(self, x: Union[float, int]) -> str:
        signed_fixed_point = self.as_signed_fixed_point(x)
        return two_complements_representation(signed_fixed_point, self.total_bit_width)


class FloatToHexFixedPointStringConverter:
    def __init__(
        self,
        total_bit_width: int,
        as_signed_fixed_point: FloatToSignedFixedPointConverter,
    ):
        self.total_bit_width = total_bit_width
        self.as_signed_fixed_point = as_signed_fixed_point

    def __call__(self, x: Union[float, int]) -> str:
        signed_fixed_point = self.as_signed_fixed_point(x)
        return hex_representation(signed_fixed_point, self.total_bit_width)


def _int_to_bin_str(number: int, bits: int) -> str:
    if number < 0:
        raise ValueError("Negative values are not supported.")
    if bits <= 0 or (number > 0 and math.log2(number) > bits):
        raise ValueError(f"The number {number} cannot be represented with {bits} bits.")
    return "{{0:0{number_of_bits}b}}".format(number_of_bits=bits).format(number)


def _int_to_hex_str(number: int, bits: int) -> str:
    if number < 0:
        raise ValueError("Negative values are not supported.")
    if bits <= 0 or (number > 0 and math.log2(number) > bits):
        raise ValueError(f"The number {number} cannot be represented with {bits} bits.")
    return 'x"{{:0{number_of_bits}x}}"'.format(
        number_of_bits=math.ceil(bits / 4)
    ).format(number)


def _get_unsigned_int_version(x, number_of_bits):
    if x < 0:
        unsigned_int_version = (1 << number_of_bits) + x
    else:
        unsigned_int_version = x
    return unsigned_int_version


def hex_representation(x: int, num_bits: int) -> str:
    unsigned_int_version = _get_unsigned_int_version(x, num_bits)
    return _int_to_hex_str(unsigned_int_version, num_bits)


def two_complements_representation(x, num_bits):
    unsigned_int_version = _get_unsigned_int_version(x, num_bits)
    return _int_to_bin_str(unsigned_int_version, num_bits)


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

    def register_symbol(self, numeric_representation: int) -> None:
        self._symbols.add(numeric_representation)
        self._update_mapping()

    def _update_mapping(self) -> None:
        sorted_numerics = list(self._symbols)
        sorted_numerics.sort()
        mapping = dict(((value, index) for index, value in enumerate(sorted_numerics)))
        self._mapping.update(mapping)

    def __len__(self):
        return len(self._symbols)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for symbol, encoded_symbol in self._mapping.values():
            yield symbol, encoded_symbol

    def __getitem__(self, item: int) -> int:
        return self._mapping[item]

    @property
    def bit_width(self) -> int:
        return math.floor(math.log(len(self._symbols), 2))

    def register_symbols(self, symbols: Iterable[int]) -> None:
        for symbol in symbols:
            self._symbols.add(symbol)
        self._update_mapping()

    def __call__(self, number: int) -> str:
        if number not in self._symbols:
            raise ValueError
        return _int_to_bin_str(self._mapping[number], self.bit_width)

    def __eq__(self, other: "ToLogicEncoder") -> bool:
        return self._symbols == other._symbols and self._mapping == other._mapping
