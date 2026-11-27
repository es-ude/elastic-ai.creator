"""Shared operations used by the layers' parameter quantization helpers."""

import torch

from ..math_operations.accumulator import accumulator_bits
from .QParams import QParams


def fold_batchnorm_affine(
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the weight and bias of the equivalent per-feature multiply-add."""
    folded_weight = weight / std
    return folded_weight, bias - folded_weight * mean


def quantize_weights(weight: torch.Tensor, qparams: QParams) -> torch.Tensor:
    """Quantize on CPU and center asymmetric weights, as in the layer helpers."""
    q_weights = qparams.quantize(weight).to("cpu")
    if not qparams.is_symmetric:
        q_weights = qparams.math_ops.intsub(
            q_weights, qparams.zero_point, int(qparams.quant_bits) + 1
        )
    return q_weights


@torch.no_grad()
def configure_bias_quantization(
    qparams: QParams,
    bias_scale: torch.Tensor,
    quant_bits: int,
) -> None:
    """Set the bias scale and range to match the accumulator."""
    qparams.set_scale_factor(bias_scale)
    qparams.set_zero_point(torch.zeros_like(qparams.zero_point))
    qparams.set_quant_range(quant_bits)
    qparams.min_float.copy_(qparams.min_quant * bias_scale)
    qparams.max_float.copy_(qparams.max_quant * bias_scale)


def configure_bias_for_accumulator(
    bias_qparams: QParams,
    input_qparams: QParams,
    weight_qparams: QParams,
    reduction_shape: tuple[int, ...] = (),
) -> None:
    """Configure fake-quantized bias from the operands of the accumulator."""
    configure_bias_quantization(
        bias_qparams,
        input_qparams.scale_factor * weight_qparams.scale_factor,
        accumulator_bits(
            input_qparams.quant_bits,
            weight_qparams.quant_bits,
            reduction_shape,
            input_scheme=input_qparams.scheme,
            weight_scheme=weight_qparams.scheme,
        ),
    )


def quantize_bias(
    bias: torch.Tensor,
    qparams: QParams,
    bias_scale: torch.Tensor,
    quant_bits: int,
) -> torch.Tensor:
    """Return bias codes at the calculated accumulator width."""
    configure_bias_quantization(qparams, bias_scale, quant_bits)
    return quantize_weights(bias, qparams)
