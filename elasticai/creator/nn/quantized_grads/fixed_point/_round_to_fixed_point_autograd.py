from typing import Any

import torch

from elasticai.creator.nn.quantized_grads.fixed_point._round_to_fixed_point import (
    quantize,
)
from elasticai.creator.nn.quantized_grads.fixed_point._two_complement_fixed_point_config import (
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
            return quantize(x, fxp_config)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        if ctx.grad_fxp_config is None:
            return *grad_outputs, None, None
        else:
            return quantize(*grad_outputs, ctx.grad_fxp_config), None, None
