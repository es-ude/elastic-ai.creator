import torch


def matmul(
    a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
) -> torch.IntTensor:
    a = a.to("cpu")
    b = b.to("cpu")
    c = a.matmul(b)
    c.clamp_(-(2 ** (c_quant_bits - 1)), (2 ** (c_quant_bits - 1)) - 1)
    return c