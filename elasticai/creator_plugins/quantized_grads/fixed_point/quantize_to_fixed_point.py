import torch
from torch import Tensor


class Round(torch.autograd.Function):
    """
    Round deterministically to nearest neighbour with STE.
    """

    @staticmethod
    def forward(ctx, x, *args, **kwargs):
        return torch.round(x)

    @staticmethod
    def backward(ctx, *grad_output):
        return grad_output


def round_tensor(x: Tensor) -> Tensor:
    return Round.apply(x)


def _clamp(
    number: Tensor, minimum_as_rational: Tensor, maximum_as_rational: Tensor
) -> Tensor:
    """
    Clamp values to twos complement.
    """
    return torch.clamp(
        number,
        minimum_as_rational,
        maximum_as_rational,
    )


def _round_to_fixed_point_hte(number: Tensor, resolution: Tensor) -> Tensor:
    """
    Implements round half to even with torch rounding function.
    """
    return round_tensor(number * resolution) / resolution


def quantize_to_fxp_hte(
    number: Tensor,
    resolution_per_int: Tensor,
    minimum_as_rational: Tensor,
    maximum_as_rational: Tensor,
) -> Tensor:
    """
    Round fixed point half to even.
    The tensor is clamped and rounded to a fixed point number.
    """
    return _clamp(
        _round_to_fixed_point_hte(number, resolution_per_int),
        minimum_as_rational,
        maximum_as_rational,
    )


def _noise(x: Tensor, resolution_per_int: Tensor) -> Tensor:
    return (torch.rand_like(x) - 0.5) / resolution_per_int


def quantize_to_fxp_stochastic(
    number: Tensor,
    resolution_per_int: Tensor,
    minimum_as_rational: Tensor,
    maximum_as_rational: Tensor,
) -> Tensor:
    return quantize_to_fxp_hte(
        number + _noise(number, resolution_per_int),
        resolution_per_int,
        minimum_as_rational,
        maximum_as_rational,
    )
