import math

from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    infer_total_and_frac_bits,
)


def pad_with_zeros(numbers: list[FixedPoint], target_length: int) -> list[FixedPoint]:
    zero = FixedPoint(0, *infer_total_and_frac_bits(numbers))
    return numbers + [zero] * (target_length - len(numbers))


def calculate_addr_width(num_items: int) -> int:
    return max(1, math.ceil(math.log2(num_items)))
