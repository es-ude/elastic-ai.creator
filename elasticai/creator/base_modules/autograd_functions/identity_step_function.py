from typing import Any

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
        inputs = inputs.cpu().to(torch.float32)
        inputs = inputs.clamp(min=step_lut.min(), max=step_lut.max())

        for step_idx in range(1, len(step_lut)):
            prev_step, curr_step = step_lut[step_idx - 1], step_lut[step_idx]
            inputs[(inputs > prev_step) & (inputs <= curr_step)] = curr_step

        return inputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None
