import math


def calculate_address_width(num_items: int) -> int:
    return max(1, math.ceil(math.log2(num_items)))
