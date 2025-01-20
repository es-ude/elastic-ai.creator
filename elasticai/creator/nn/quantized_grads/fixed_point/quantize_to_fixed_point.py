import torch

from .two_complement_fixed_point_config import FixedPointConfigV2


def _clamp_(number: torch.Tensor, fxp_conf: FixedPointConfigV2) -> None:
    """
    Inplace clamp values to twos complement.
    """
    number.clamp_(
        fxp_conf.minimum_as_rational_tensor, fxp_conf.maximum_as_rational_tensor
    )


def _clamp(number: torch.Tensor, fxp_conf: FixedPointConfigV2) -> torch.Tensor:
    """
    Clamp values to twos complement.
    """
    return torch.clamp(
        number,
        fxp_conf.minimum_as_rational_tensor,
        fxp_conf.maximum_as_rational_tensor,
    )


def _round_to_fixed_point_hte(
    number: torch.Tensor, resolution: torch.Tensor
) -> torch.Tensor:
    """
    Implements round half to even with torch rounding function.
    """
    return torch.round(number * resolution) / resolution


def _round_to_fixed_point_hte_(
    number: torch.Tensor, fxp_resolution: torch.Tensor
) -> None:
    """
    Implements round half to even with torch rounding function. Inplace operation.
    """
    number.mul_(fxp_resolution).round_().div_(fxp_resolution)


def quantize_to_fxp_stochastic(
    number: torch.Tensor, fxp_conf: FixedPointConfigV2
) -> torch.Tensor:
    """
    Round fixed point stochastic adds a noise of [-0.5/2**fracbits to 0.5/2**fracbits] on the input tensor.
    The tensor is clamped and rounded to a fixed point number.
    """
    noise = (
        torch.rand_like(number, device=fxp_conf.device) - 0.5
    ) / fxp_conf.resolution_per_int
    return _round_to_fixed_point_hte(
        _clamp(number, fxp_conf) + noise, fxp_conf.resolution_per_int
    )


def quantize_to_fxp_stochastic_(
    number: torch.Tensor, fxp_conf: FixedPointConfigV2
) -> None:
    """
    Inplace operation of quantize_to_fxp_stochastic.
    Round fixed point stochastic adds a noise of [-0.5/2**fracbits to 0.5/2**fracbits] on the input tensor.
    The tensor is clamped and rounded to a fixed point number.
    """
    noise = (
        torch.rand_like(number, device=fxp_conf.device) - 0.5
    ) / fxp_conf.resolution_per_int
    _clamp_(number, fxp_conf)
    number.add_(noise)
    _round_to_fixed_point_hte_(number, fxp_conf.resolution_per_int)


def quantize_to_fxp_hte(
    number: torch.Tensor, fxp_conf: FixedPointConfigV2
) -> torch.Tensor:
    """
    Round fixed point half to even.
    The tensor is clamped and rounded to a fixed point number.
    """
    return _clamp(
        _round_to_fixed_point_hte(number, fxp_conf.resolution_per_int), fxp_conf
    )


def quantize_to_fxp_hte_(number: torch.Tensor, fxp_conf: FixedPointConfigV2) -> None:
    """
    Inplace operation of quantize_to_fxp_hte.
    Round fixed point half to even.
    The tensor is clamped and rounded to a fixed point number.
    """
    _round_to_fixed_point_hte_(number, fxp_conf.resolution_per_int)
    _clamp_(number, fxp_conf)
