import torch


def calculate_asymmetric_quant_params(
    min_float: torch.FloatTensor,
    max_float: torch.FloatTensor,
    min_quant: torch.IntTensor,
    max_quant: torch.IntTensor,
    eps: torch.FloatTensor,
):
    scale_factor = (max_float - min_float) / (max_quant.float() - min_quant.float())
    scale_factor = torch.max(scale_factor, eps)

    zero_point = max_quant - (max_float / scale_factor)
    zero_point = zero_point.round_().clamp(min_quant, max_quant)

    return scale_factor, zero_point, min_float, max_float


def calculate_symmetric_quant_params(
    min_quant: torch.IntTensor,
    max_quant: torch.IntTensor,
    min_float: torch.FloatTensor,
    max_float: torch.FloatTensor,
    eps: torch.FloatTensor,
):
    max_extent = torch.max(torch.abs(min_float), torch.abs(max_float))
    max_float = max_extent
    min_float = -max_extent

    scale_factor = (max_float - min_float) / (max_quant.float() - min_quant.float())
    scale_factor = torch.max(scale_factor, eps)

    zero_point = torch.zeros(scale_factor.size())

    return scale_factor, zero_point, min_float, max_float
