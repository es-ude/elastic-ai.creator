from typing import Any, Callable

import torch


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
        forward_quantize: Callable[[torch.Tensor], torch.Tensor] = args[1]
        return forward_quantize(x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None


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
        forward_quantize: Callable[[torch.Tensor], torch.Tensor] = args[1]
        ctx.backward_quantize: Callable[[torch.Tensor], torch.Tensor] = args[2]
        return forward_quantize(x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return ctx.backward_quantize(*grad_outputs), None, None
