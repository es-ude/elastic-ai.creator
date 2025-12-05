import torch
from torch import Tensor

from elasticai.creator_plugins.quantized_grads.round import Round, _noise


def quantize_to_int_hte_fake(number: Tensor, min_value: Tensor, max_value: Tensor) -> Tensor:
    return Round.apply(torch.clamp(number, min_value, max_value))

def quantize_to_int_stochastic_fake(number: Tensor, min_value: Tensor, max_value: Tensor) -> Tensor:
    noise = _noise(number)
    return quantize_to_int_hte_fake(number+noise, min_value, max_value)