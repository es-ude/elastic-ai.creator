import torch


def subtract(
    a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
) -> torch.IntTensor:
    c = a - b
    c.clamp_(-(2 ** (c_quant_bits - 1)), (2 ** (c_quant_bits - 1)) - 1)
    return c
