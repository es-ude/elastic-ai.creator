from typing import cast

import torch

from elasticai.creator.nn.fixed_point._math_operations import MathOperations
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.vhdl.shared_designs.precomputed_scalar_function import (
    PrecomputedScalarFunction,
)
from elasticai.creator.vhdl.translatable import Translatable

from .identity_step_function import IdentityStepFunction


class PrecomputedModule(torch.nn.Module, Translatable):
    def __init__(
        self,
        base_module: torch.nn.Module,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float],
    ) -> None:
        super().__init__()
        self._base_module = base_module
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
        self._operations = MathOperations(self._config)
        self._step_lut = torch.linspace(*sampling_intervall, num_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._stepped_inputs(x)
        outputs = self._base_module(x)
        return self._operations.quantize(outputs)

    def translate(self, name: str) -> PrecomputedScalarFunction:
        quantized_inputs = list(map(self._config.as_integer, self._step_lut.tolist()))
        return PrecomputedScalarFunction(
            name=name,
            width=self._config.total_bits,
            inputs=quantized_inputs,
            function=self._quantized_inference,
        )

    def _stepped_inputs(self, x: torch.Tensor) -> torch.Tensor:
        step_inputs = cast(torch.Tensor, IdentityStepFunction.apply(x, self._step_lut))
        return self._operations.quantize(step_inputs)

    def _quantized_inference(self, x: int) -> int:
        fxp_input = self._config.as_rational(x)
        with torch.no_grad():
            output = self(torch.tensor(fxp_input))
        return self._config.as_integer(float(output.item()))
