import torch
from torch import Tensor

from elasticai.creator_plugins.quantized_grads.round import Round, _noise


def quantize_to_int_hte_fake(number: Tensor, min_value: Tensor, max_value: Tensor) -> Tensor:
    return Round.apply(torch.clamp(number, min_value, max_value))

def quantize_to_int_stochastic_fake(number: Tensor, min_value: Tensor, max_value: Tensor) -> Tensor:
    noise = _noise(number)
    return quantize_to_int_hte_fake(number+noise, min_value, max_value)

def quantize_to_int_hte(number: Tensor, min_value: Tensor, max_value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return quantize_to_int_hte_fake(number, min_value, max_value), Tensor([1.]).to(number.device), Tensor([0.]).to(number.device)

def quantize_to_int_stochastic(number: Tensor, min_value: Tensor, max_value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return quantize_to_int_hte(number+_noise(number), min_value, max_value)