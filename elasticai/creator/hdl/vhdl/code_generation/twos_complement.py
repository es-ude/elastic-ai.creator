def to_unsigned(value: int, total_bits: int) -> int:
    def invert(value: int) -> int:
        return value ^ int("1" * total_bits, 2)

    def discard_leading_bits(value: int) -> int:
        return value & int("1" * total_bits, 2)

    if value < 0:
        value = discard_leading_bits(invert(abs(value)) + 1)

    return value
