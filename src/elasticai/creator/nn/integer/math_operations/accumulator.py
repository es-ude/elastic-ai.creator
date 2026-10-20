"""Width calculation shared by integer accumulation operations."""

import torch


def _calculate_bits_for_mac_op(product_bits: int, num_products: int = 1):
    if num_products < 1:
        raise ValueError("An accumulator must contain at least one product")
    mac_bits = product_bits + (num_products - 1).bit_length()
    return mac_bits


def _calculate_bits_for_bias_addition(mac_bits: int, bias: torch.Tensor | None) -> int:
    if bias is None:
        return mac_bits
    else:
        max_bias = max(abs(int(bias.min().item())), abs(int(bias.max().item())))
        bias_bits = max(2, max_bias.bit_length() + 1)
        accumulator_bits = max(mac_bits, bias_bits) + 1
        return accumulator_bits


def accumulator_bits(
    product_bits: int, num_products: int = 1, bias: torch.Tensor | None = None
) -> int:
    """Allow for the reduction and for adding the actual quantized bias."""
    mac_bits = _calculate_bits_for_mac_op(product_bits, num_products)
    accumulator_bits = _calculate_bits_for_bias_addition(mac_bits, bias)
    if accumulator_bits > 64:
        raise ValueError("The required accumulator width exceeds int64")
    return accumulator_bits
