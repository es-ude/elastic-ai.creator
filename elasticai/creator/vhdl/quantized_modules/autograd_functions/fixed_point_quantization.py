from typing import Any

import torch

from elasticai.creator.vhdl.number_representations import (
    FixedPointFactory,
    fixed_point_params_from_factory,
)


class FixedPointQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, fixed_point_factory: FixedPointFactory
    ) -> torch.Tensor:
        total_bits, frac_bits = fixed_point_params_from_factory(fixed_point_factory)
        largest_fp_int = 2 ** (total_bits - 1) - 1
        return (
            (x * (1 << frac_bits)).int().float().clamp(-largest_fp_int, largest_fp_int)
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None


class FixedPointDequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, fixed_point_factory: FixedPointFactory
    ) -> torch.Tensor:
        total_bits, frac_bits = fixed_point_params_from_factory(fixed_point_factory)
        min_value = 2 ** (total_bits - frac_bits - 1) * (-1)
        max_value = (2 ** (total_bits - 1) - 1) / (1 << frac_bits)
        return (x / (1 << frac_bits)).clamp(min_value, max_value)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None
