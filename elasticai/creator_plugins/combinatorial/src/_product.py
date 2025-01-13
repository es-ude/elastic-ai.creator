from typing import Iterable


def _product(numbers: Iterable[int]) -> int:
    prod = 1
    for num in numbers:
        prod *= num
    return prod
