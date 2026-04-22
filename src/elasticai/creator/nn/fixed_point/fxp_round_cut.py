from typing import Any

import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
)


class RoundToFixedPoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) != 2:
            raise TypeError(
                "apply() takes exactly two arguments "
                "(x: torch.Tensor, config: FixedPointConfig)"
            )
        x: torch.Tensor = args[0]
        config: FxpArithmetic = args[1]

        fxp_ints = config.round_to_integer(x)
        out_of_bounds = fxp_ints[config.integer_out_of_bounds(fxp_ints)]
        if torch.any(out_of_bounds):
            raise ValueError("Cannot quantize tensor. Values out of bounds.")

        return config.as_rational(fxp_ints)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None


class CutToFixedPoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) != 2:
            raise TypeError(
                "apply() takes exactly two arguments "
                "(x: torch.Tensor, config: FixedPointConfig)"
            )
        x: torch.Tensor = args[0]
        config: FxpArithmetic = args[1]

        fxp_ints = config.cut_as_integer(x)
        out_of_bounds = config.integer_out_of_bounds(fxp_ints)
        if torch.any(out_of_bounds):
            raise ValueError("Cannot quantize tensor. Values out of bounds.")

        return config.as_rational(fxp_ints)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None
