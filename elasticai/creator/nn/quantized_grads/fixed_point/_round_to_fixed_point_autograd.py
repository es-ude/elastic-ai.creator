from typing import Any, Callable

import torch

from ._two_complement_fixed_point_config import FixedPointConfigV2
from .quantize_to_fixed_point import quantize_to_fxp_hte, quantize_to_fxp_stochastic


def _make_autograd_function_forw(
    _quantize_to_fxp: Callable[[torch.Tensor, FixedPointConfigV2], torch.Tensor],
) -> type[torch.autograd.Function]:
    class QuantizeForw(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
            if len(args) != 2:
                raise TypeError(
                    "apply() takes exactly two arguments "
                    "(x: torch.Tensor, config: FixedPointConfig)"
                )
            if len(kwargs) != 0:
                raise TypeError(
                    f"apply() takes exactly two arguments "
                    f"(x: torch.Tensor, config: FixedPointConfig)"
                    f"You provided {len(kwargs)=} arguments. But should provide 0"
                )
            x: torch.Tensor = args[0]
            forward_fxp_config: FixedPointConfigV2 = args[1]
            return _quantize_to_fxp(x, forward_fxp_config)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            return *grad_outputs, None

    return QuantizeForw


def _make_autograd_function_forwbackw(
    _quantize_to_fxp: Callable[[torch.Tensor, FixedPointConfigV2], torch.Tensor],
) -> type[torch.autograd.Function]:
    class QuantizeForwBackw(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
            if len(args) != 3:
                raise TypeError(
                    "apply() takes exactly three arguments "
                    "(x: torch.Tensor, config: FixedPointConfig)"
                )
            if len(kwargs) != 0:
                raise TypeError(
                    f"apply() takes exactly three arguments "
                    f"(x: torch.Tensor, forward_config: FixedPointConfigV2, backward_config: FixedPointConfigV2)"
                    f"You provided {len(kwargs)=} arguments. But should provide 0"
                )
            x: torch.Tensor = args[0]
            forward_config: FixedPointConfigV2 = args[1]
            ctx.back_config = args[2]
            return _quantize_to_fxp(x, forward_config)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            return _quantize_to_fxp(*grad_outputs, ctx.back_config), None, None

    return QuantizeForwBackw


QuantizeForwStochastic = _make_autograd_function_forw(quantize_to_fxp_stochastic)
QuantizeForwHTE = _make_autograd_function_forw(quantize_to_fxp_hte)

QuantizeForwBackwStochastic = _make_autograd_function_forwbackw(
    quantize_to_fxp_stochastic
)
QuantizeForwBackwHTE = _make_autograd_function_forwbackw(quantize_to_fxp_hte)
