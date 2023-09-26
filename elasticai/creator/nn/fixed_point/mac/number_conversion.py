"""
Here we collect several functions to convert fixed point,
integer and natural numbers to bit patterns and vice versa.

IMPORTANT: We assume, the numbers to be representable in
the target format!
"""


def _toggle_bits(number: int, total_bits: int) -> int:
    def invert(value: int) -> int:
        return value ^ int("1" * total_bits, 2)

    def discard_leading_bits(value: int) -> int:
        return value & int("1" * total_bits, 2)

    return discard_leading_bits(invert(abs(number)))


def _twos_complement(number, total_bits):
    return _toggle_bits(number, total_bits) + 1


def bits_to_integer(pattern: str) -> int:
    is_negative = pattern[0] == "1"
    number = int(pattern, 2)
    if is_negative:
        number = -_twos_complement(number, len(pattern))
    return number


def bits_to_rational(pattern: str, frac_bits: int) -> float:
    pattern = pattern.strip()
    number = bits_to_integer(pattern)
    return number / (1 << frac_bits)


def convert_rational_to_bit_pattern(
    rational: float, total_bits: int, frac_bits: int
) -> str:
    return integer_to_bits(int(rational * (1 << frac_bits)), total_bits=total_bits)


def bits_to_natural(pattern: str) -> int:
    return int(pattern, 2)


def integer_to_bits(number: int, total_bits: int) -> str:
    if number < 0:
        number = _twos_complement(number, total_bits)
    return f"{number:0{total_bits}b}"


def _max_twos_complement_pattern(total_bits):
    return "0" + "1" * (total_bits - 1)


def _min_twos_complement_pattern(total_bits):
    return "1" + "0" * (total_bits - 1)


def max_rational(total_bits: int, frac_bits: int) -> float:
    return bits_to_rational(
        _max_twos_complement_pattern(total_bits), frac_bits=frac_bits
    )


def min_rational(total_bits: int, frac_bits: int) -> float:
    return bits_to_rational(
        _min_twos_complement_pattern(total_bits), frac_bits=frac_bits
    )


def min_integer(total_bits: int) -> int:
    return bits_to_integer(_min_twos_complement_pattern(total_bits))


def max_integer(total_bits: int) -> int:
    return bits_to_integer(_max_twos_complement_pattern(total_bits))


def min_natural(total_bits: int) -> int:
    return 0


def max_natural(total_bits: int) -> int:
    return bits_to_natural("1" * total_bits)
