import torch

from ._two_complement_fixed_point_config import FixedPointConfigV2


def quantize(number: torch.Tensor, fxp_conf: FixedPointConfigV2) -> torch.Tensor:
    return _round(clamp(number, fxp_conf), fxp_conf)


def _round(number: torch.Tensor, fxp_conf: FixedPointConfigV2) -> torch.Tensor:
    if fxp_conf.stochastic_rounding:
        noise = (torch.rand_like(number) - 0.5) / (2**fxp_conf.frac_bits)
        return round_to_fixed_point(number + noise, fxp_conf.frac_bits)
    else:
        return round_to_fixed_point(number, fxp_conf.frac_bits)


def clamp(number: torch.Tensor, fxp_conf: FixedPointConfigV2) -> torch.Tensor:
    return torch.clamp(
        number,
        fxp_conf.minimum_as_rational_tensor,
        fxp_conf.maximum_as_rational_tensor,
    )


def round_to_fixed_point(number: torch.Tensor, frac_bits: int) -> torch.Tensor:
    return torch.round(number * (2**frac_bits)) / 2**frac_bits
