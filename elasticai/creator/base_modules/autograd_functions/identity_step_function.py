from typing import Any, cast

import torch


class IdentityStepFunction(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        raise NotImplementedError()

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) != 2:
            raise TypeError(
                "apply() takes exactly two arguments "
                "(inputs: torch.Tensor, step_lut: torch.Tensor)"
            )

        inputs: torch.Tensor = args[0]
        step_lut: torch.Tensor = args[1]

        steps = len(step_lut)
        if steps < 2:
            raise ValueError(
                f"Number of steps cannot be less than or equal to 1 (steps == {steps})."
            )

        inputs = inputs.to(torch.float32)
        clipped_inputs = inputs.clamp(min=step_lut.min(), max=step_lut.max())

        step_lut, _ = torch.sort(step_lut, descending=True)

        def get_step_value(input_value: torch.Tensor) -> torch.Tensor:
            return step_lut[int(sum(step_lut >= input_value) - 1)]

        return clipped_inputs.detach().apply_(get_step_value)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None
