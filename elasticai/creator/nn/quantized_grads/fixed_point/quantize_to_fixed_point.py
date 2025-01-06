import torch

from elasticai.creator.nn.quantized_grads.fixed_point import FixedPointConfigV2


def quantize_to_fxp_stochastic(
    number: torch.Tensor, fxp_conf: FixedPointConfigV2
) -> torch.Tensor:
    """
    Round fixed point stochastic adds a noise of [-0.5/2**fracbits to 0.5/2**fracbits] on the input tensor.
    The tensor is clamped and rounded to a fixed point number.
    """
    noise = (torch.rand_like(number) - 0.5) / (2**fxp_conf.frac_bits)
    return _round_to_fixed_point_hte(
        _clamp(number, fxp_conf) + noise, fxp_conf.frac_bits
    )


def quantize_to_fxp_hte(
    number: torch.Tensor, fxp_conf: FixedPointConfigV2
) -> torch.Tensor:
    """
    Round fixed point half to even.
    The tensor is clamped and rounded to a fixed point number.
    """
    return _clamp(_round_to_fixed_point_hte(number, fxp_conf.frac_bits), fxp_conf)


def _clamp(number: torch.Tensor, fxp_conf: FixedPointConfigV2) -> torch.Tensor:
    return torch.clamp(
        number,
        fxp_conf.minimum_as_rational_tensor,
        fxp_conf.maximum_as_rational_tensor,
    )


def _round_to_fixed_point_hte(number: torch.Tensor, frac_bits: int) -> torch.Tensor:
    """
    Implements round half to even with torch rounding function
    """
    return torch.round(number * (2**frac_bits)) / 2**frac_bits
