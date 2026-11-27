"""Bitwidth configuration for standard integer layers."""


def split_quant_bits(quant_bits: int | tuple[int, int]) -> tuple[int, int]:
    """Return the weight and activation widths, in that order."""
    if isinstance(quant_bits, int):
        return quant_bits, quant_bits
    if (
        isinstance(quant_bits, tuple)
        and len(quant_bits) == 2
        and all(isinstance(width, int) for width in quant_bits)
    ):
        return quant_bits
    raise TypeError("quant_bits must be an int or a tuple (weight_bits, activation_bits)")


def require_uniform_quant_bits(weight_bits: int, activation_bits: int) -> None:
    """Check the single-width requirement of the standard integer VHDL designs."""
    if weight_bits != activation_bits:
        raise ValueError(
            "VHDL design requires equal weight and activation bit widths "
            f"(got {weight_bits} for weights and {activation_bits} for activations)"
        )
