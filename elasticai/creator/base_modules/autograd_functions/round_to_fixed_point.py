from typing import Any

import torch

from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)


class RoundToFixedPoint(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        raise NotImplementedError()

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) != 2:
            raise TypeError(
                "apply() takes exactly two arguments "
                "(x: torch.Tensor, config: FixedPointConfig)"
            )
        x: torch.Tensor = args[0]
        config: FixedPointConfig = args[1]

        fp_ints = config.as_integer(x)
        out_of_bounds = fp_ints[config.integer_out_of_bounds(fp_ints)]
        if torch.any(out_of_bounds):
            raise ValueError("Cannot quantize tensor. Values out of bounds.")

        return config.as_rational(fp_ints)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None
