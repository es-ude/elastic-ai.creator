def _toggle_bits_for_negative_number(number: int, total_bits: int) -> int:
    return -(number + int("1" * total_bits, 2))


def _toggle_bits_for_positive_number(number: int, total_bits: int) -> int:
    return -_toggle_bits_for_negative_number(-number, total_bits)


def _toggle_bits(number: int, total_bits: int) -> int:
    if number < 0:
        return _toggle_bits_for_negative_number(number, total_bits)
    return _toggle_bits_for_positive_number(number, total_bits)


def bits_to_integer(pattern: str) -> int:
    is_negative = pattern[0] == "1"
    number = int(pattern, 2)
    if is_negative:
        number = _toggle_bits(number, len(pattern)) + 1
        number = -number
    return number


def bits_to_rational(pattern: str, frac_bits: int) -> float:
    number = bits_to_integer(pattern)
    return number / (1 << frac_bits)


def bits_to_natural(pattern: str) -> int:
    return int(pattern, 2)


def integer_to_bits(number: int, total_bits: int) -> str:
    if number < 0:
        number = -_toggle_bits(number, total_bits) + 1
    return f"{number:0{total_bits}b}"
