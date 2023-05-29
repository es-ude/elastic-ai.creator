from typing import cast

import torch

from elasticai.creator.base_modules.arithmetics import Arithmetics
from elasticai.creator.base_modules.autograd_functions.step_function_inputs import (
    StepFunctionInputs,
)


class Tanh(torch.nn.Tanh):
    def __init__(
        self,
        arithmetics: Arithmetics,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-5, 5),
    ) -> None:
        super().__init__()
        self._arithmetics = arithmetics
        self.num_steps = num_steps
        self.sampling_intervall = sampling_intervall

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        step_inputs = cast(
            torch.Tensor,
            StepFunctionInputs.apply(inputs, *self.sampling_intervall, self.num_steps),
        )
        quantized_step_inputs = self._arithmetics.quantize(step_inputs)
        outputs = torch.nn.functional.tanh(quantized_step_inputs)
        return self._arithmetics.quantize(outputs)
