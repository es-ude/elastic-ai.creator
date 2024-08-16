import torch


def quantizeProcess(
    x: torch.FloatTensor,
    min_quant: torch.IntTensor,
    max_quant: torch.IntTensor,
    scale_factor: torch.FloatTensor,
    zero_point: torch.IntTensor,
) -> torch.IntTensor:
    "compatiable with symmetric/asymmetric, signed/unsignd quantization"
    x_q = x / scale_factor + zero_point
    x_q = x_q.round_().clamp(min=min_quant.item(), max=max_quant.item())
    x_q = x_q.to(torch.int32)

    return x_q
