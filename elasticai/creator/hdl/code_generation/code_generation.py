import math


def calculate_address_width(num_items: int) -> int:
    return max(1, math.ceil(math.log2(num_items)))


def to_hex(number: int, bit_width: int) -> str:
    return f"{number:0{math.ceil(bit_width / 4)}x}"
