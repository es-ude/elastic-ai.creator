import torch


def intadd(a: torch.Tensor, b: torch.Tensor, c_quant_bits: int) -> torch.Tensor:
    """
    Add two integers represented as torch.Tensors and get the integer result.
    Only support signed integers with specific quantization bits.
    """

    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("Inputs of addition must be torch.Tensors")

    if a.dtype != torch.int32 or b.dtype != torch.int32:
        raise TypeError("Input tensors are not torch.IntTensor (torch.int32)")

    c_MAX = 2 ** (c_quant_bits - 1) - 1
    c_MIN = -(2 ** (c_quant_bits - 1))

    c = a + b
    c = torch.clamp(c, c_MIN, c_MAX)

    return c
