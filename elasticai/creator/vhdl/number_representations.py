from typing import Union, overload


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
        if not x_tmp.is_integer() and self._strict:
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


def two_complements_representation(x, number_of_bits):
    if x < 0:
        unsigned_int_version = (1 << number_of_bits) + x
    else:
        unsigned_int_version = x
    return "{{0:0{number_of_bits}b}}".format(number_of_bits=number_of_bits).format(
        unsigned_int_version
    )
