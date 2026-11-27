"""Width calculation shared by integer accumulation operations."""

from __future__ import annotations

import warnings
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..quant_utils.QParams import QuantScheme

MAX_ACCUMULATOR_BITS = 32


def accumulator_bits(
    input_bits: int,
    weight_bits: int,
    reduction_shape: tuple[int, ...] = (),
    *,
    input_scheme: QuantScheme | None = None,
    weight_scheme: QuantScheme | None = None,
) -> int:
    """Size a signed accumulator, assuming asymmetric operands where unspecified."""
    num_products = prod(reduction_shape)
    if num_products < 1:
        raise ValueError("An accumulator must contain at least one product")

    schemes = (input_scheme, weight_scheme)
    centering_bits = sum(
        scheme is None or not scheme.is_symmetric for scheme in schemes
    )
    # A product of two unsigned, zero-centered operands needs room for a sign
    # in the signed accumulator.
    unsigned_product = all(
        scheme is not None and scheme.is_symmetric and not scheme.is_signed
        for scheme in schemes
    )
    product_bits = int(input_bits) + int(weight_bits) + centering_bits
    return _cap_accumulator_bits(
        product_bits + (num_products - 1).bit_length() + int(unsigned_product)
    )


def _cap_accumulator_bits(bits: int) -> int:
    if bits > MAX_ACCUMULATOR_BITS:
        warnings.warn(
            f"The calculated accumulator width is {bits} bits; "
            f"using {MAX_ACCUMULATOR_BITS} bits instead. Overflow could occur.",
            RuntimeWarning,
            stacklevel=3,
        )
        return MAX_ACCUMULATOR_BITS
    return bits
