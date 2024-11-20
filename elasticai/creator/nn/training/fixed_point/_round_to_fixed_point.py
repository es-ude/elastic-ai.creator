from typing import Any

import torch

from elasticai.creator.nn.training.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfigV2,
)


class RoundToFixedPointTrainable(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) != 3:
            raise TypeError(
                "apply() takes exactly two arguments "
                "(x: torch.Tensor, config: FixedPointConfig)"
            )
        x: torch.Tensor = args[0]
        fxp_config: FixedPointConfigV2 | None = args[1]
        ctx.grad_fxp_config: FixedPointConfigV2 | None = args[2]
        if fxp_config is None:
            return x
        else:
            return fxp_config.quantize(x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        if ctx.grad_fxp_config is None:
            return *grad_outputs, None, None
        else:
            return ctx.grad_fxp_config.quantize(*grad_outputs), None, None
