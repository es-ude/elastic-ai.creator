import torch
from torch import Tensor

from elasticai.creator_plugins.quantized_grads.round import Round, _noise


def quantize_linear_hte(number: Tensor, min_value: Tensor, max_value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    min = torch.min(number)
    max = torch.max(number)
    scale = (max-min)/(max_value)
    zero_point = Round.apply(min/scale)

    #zero_point = torch.floor(min/(max-min)*max_value)
    #scale = (max-zero_point)/max_value
    norm = number/scale - zero_point
    return Round.apply(torch.clamp(norm, min_value, max_value)), scale, zero_point

def quantize_linear_stochastic(number: Tensor, min_value: Tensor, max_value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return quantize_linear_hte(number+_noise(number), min_value, max_value)

def quantize_linear_hte_fake(number: Tensor, min_value: Tensor, max_value: Tensor) -> Tensor:
    number, scale, zero_point = quantize_linear_hte(number, min_value, max_value)
    return dequantize_linear(number, scale, zero_point)

def quantize_linear_stochastic_fake(number: Tensor, min_value: Tensor, max_value: Tensor) -> Tensor:
    number, scale, zero_point = quantize_linear_stochastic(number, min_value, max_value)
    return dequantize_linear(number, scale, zero_point)

def dequantize_linear(number: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
    return (number + zero_point) * scale