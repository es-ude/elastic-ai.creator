from typing import Any, cast

import torch


class StepFunctionInputs(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        raise NotImplementedError()

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) != 4:
            raise TypeError(
                "apply() takes exactly four arguments "
                "(inputs: torch.Tensor, minimum: float, maximum: float, steps: int)"
            )
        inputs: torch.Tensor = args[0]
        minimum, maximum, steps = cast(tuple[float, float, int], args[1:4])

        if steps < 2:
            raise ValueError(
                f"Number of steps cannot be less than or equal to 1 (steps == {steps})."
            )

        input_lut = torch.linspace(minimum, maximum, steps).flip(dims=[0])
        clipped_inputs = inputs.clamp(min=minimum, max=maximum)
        outputs = clipped_inputs.detach().apply_(
            lambda x: input_lut[int(sum(input_lut >= x) - 1)]
        )

        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return *grad_outputs, None
