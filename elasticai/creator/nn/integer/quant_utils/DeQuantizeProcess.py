import torch

from elasticai.creator.nn.integer.math_operations.subtraction import subtract


def dequantizeProcess(
    x_q: torch.IntTensor,
    scale: torch.FloatTensor,
    zero_point: torch.IntTensor,
    quant_bits: int,
) -> torch.FloatTensor:
    "compatiable with symmetric/asymmetric, signed/unsignd de-quantization"

    x_q = subtract(x_q, zero_point, quant_bits)
    x_dq = x_q.to(torch.float32) * scale
    return x_dq
