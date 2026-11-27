"""Shared operations used by the layers' parameter quantization helpers."""

from dataclasses import dataclass

import torch

from .._dtypes import FLOAT_DTYPES, check_dtype
from ..math_operations.accumulator import accumulator_bits
from .QParams import QParams, _quantize_tensor


@dataclass(frozen=True)
class AccumulatorQParams:
    """Observer-free signed quantization parameters with a zero point of zero."""

    quant_bits: int
    scale_factor: torch.Tensor

    @property
    def _limit(self) -> int:
        return (1 << (self.quant_bits - 1)) - 1

    @property
    def min_float(self) -> torch.Tensor:
        return -self._limit * self.scale_factor

    @property
    def max_float(self) -> torch.Tensor:
        return self._limit * self.scale_factor

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        check_dtype("x", x, *FLOAT_DTYPES)
        return _quantize_tensor(
            x, self.scale_factor.to(x.device), 0, -self._limit, self._limit
        )

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        return (codes.to(torch.float32) * self.scale_factor.to(codes.device)).to(
            torch.float32
        )


def fold_batchnorm_affine(
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the weight and bias of the equivalent per-feature multiply-add."""
    folded_weight = weight / std
    return folded_weight, bias - folded_weight * mean


def quantize_weights(
    weight: torch.Tensor, qparams: QParams | AccumulatorQParams
) -> torch.Tensor:
    """Quantize on CPU and center asymmetric weights, as in the layer helpers."""
    q_weights = qparams.quantize(weight).to("cpu")
    if not isinstance(qparams, AccumulatorQParams) and not qparams.is_symmetric:
        q_weights = qparams.math_ops.intsub(
            q_weights, qparams.zero_point, int(qparams.quant_bits) + 1
        )
    return q_weights


def accumulator_qparams(
    input_qparams: QParams,
    weight_qparams: QParams,
    reduction_shape: tuple[int, ...] = (),
) -> AccumulatorQParams:
    """Derive the bias scale and width from the operands of the accumulator."""
    return AccumulatorQParams(
        quant_bits=accumulator_bits(
            input_qparams.quant_bits,
            weight_qparams.quant_bits,
            reduction_shape,
            input_scheme=input_qparams.scheme,
            weight_scheme=weight_qparams.scheme,
        ),
        scale_factor=input_qparams.scale_factor * weight_qparams.scale_factor,
    )


def quantize_bias(bias: torch.Tensor, accumulator: AccumulatorQParams) -> torch.Tensor:
    """Return bias codes at the calculated accumulator width."""
    return quantize_weights(bias, accumulator)
