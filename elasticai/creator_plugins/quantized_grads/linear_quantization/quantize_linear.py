import torch
from torch import Tensor

from elasticai.creator_plugins.quantized_grads.round import Round, _noise, round_tensor


def quantize_linear_asym_hte(number: Tensor, min_value: Tensor, max_value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    min_number = torch.min(number)
    max_number = torch.max(number)
    scale = (max_number-min_number)/(max_value)
    zero_point = round_tensor(min_number/scale)
    norm = number/scale - zero_point
    return Round.apply(torch.clamp(norm, min_value, max_value)), scale, zero_point

def quantize_linear_asym_stochastic(number: Tensor, min_value: Tensor, max_value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return quantize_linear_asym_hte(number + _noise(number), min_value, max_value)

def quantize_linear_asym_hte_fake(number: Tensor, min_value: Tensor, max_value: Tensor) -> Tensor:
    number, scale, zero_point = quantize_linear_asym_hte(number, min_value, max_value)
    return dequantize_linear(number, scale, zero_point)

def quantize_linear_asym_stochastic_fake(number: Tensor, min_value: Tensor, max_value: Tensor) -> Tensor:
    number, scale, zero_point = quantize_linear_asym_stochastic(number, min_value, max_value)
    return dequantize_linear(number, scale, zero_point)

def dequantize_linear(number: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
    return (number + zero_point) * scale

