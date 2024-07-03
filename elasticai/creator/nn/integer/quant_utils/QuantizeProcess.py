import torch


def quantizeProcess(
    x_r: float, min_quant: int, max_quant: int, scale: float, zero_point: int
) -> int:
    "compatiable with symmetric/asymmetric, signed/unsignd quantization"
    x_r = x_r.to(scale.device)
    x_q = x_r / scale + zero_point
    x_q = x_q.round_().clamp(min=min_quant, max=max_quant)
    x_q_int = x_q.to(torch.int32)
    return x_q_int
