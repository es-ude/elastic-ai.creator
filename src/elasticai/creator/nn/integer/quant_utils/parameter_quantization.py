"""Shared operations used by the layers' parameter quantization helpers."""

import torch

from .MPQParams import QParams as MPQParams
from .QParams import QParams


def quantize_weights(
    weight: torch.Tensor, qparams: QParams | MPQParams
) -> torch.Tensor:
    """Quantize on CPU and center asymmetric weights, as in the layer helpers."""
    q_weights = qparams.quantize(weight).to("cpu")
    if not qparams.is_symmetric:
        q_weights = qparams.math_ops.intsub(
            q_weights, qparams.zero_point, int(qparams.quant_bits) + 1
        )
    return q_weights


def quantize_bias(
    bias: torch.Tensor,
    qparams: QParams | MPQParams,
    accumulator_scale: torch.Tensor,
    quant_bits: int,
) -> torch.Tensor:
    """Quantize bias using the layer's chosen scale and width; return CPU codes."""
    qparams.set_scale_factor(accumulator_scale)
    qparams.set_zero_point(torch.zeros_like(qparams.zero_point))
    qparams.set_quant_range(quant_bits)
    return quantize_weights(bias, qparams)
