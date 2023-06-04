from typing import cast

import torch

from elasticai.creator.base_modules.arithmetics import Arithmetics
from elasticai.creator.base_modules.autograd_functions.identity_step_function import (
    IdentityStepFunction,
)


class Tanh(torch.nn.Tanh):
    def __init__(
        self,
        arithmetics: Arithmetics,
        num_steps: int,
        sampling_intervall: tuple[float, float],
    ) -> None:
        super().__init__()
        self._arithmetics = arithmetics
        self._step_lut = torch.linspace(*sampling_intervall, num_steps)

    def _quantized_step_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        step_inputs = cast(
            torch.Tensor, IdentityStepFunction.apply(inputs, self._step_lut)
        )
        return self._arithmetics.quantize(step_inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self._quantized_step_inputs(inputs)
        outputs = torch.nn.functional.tanh(inputs)
        return self._arithmetics.quantize(outputs)
