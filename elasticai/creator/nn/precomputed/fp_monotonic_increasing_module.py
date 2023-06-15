from typing import cast

import torch

from elasticai.creator.base_modules.arithmetics.fixed_point_arithmetics import (
    FixedPointArithmetics,
)
from elasticai.creator.base_modules.autograd_functions.identity_step_function import (
    IdentityStepFunction,
)
from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.vhdl.design_base.design import Design
from elasticai.creator.vhdl.designs.precomputed_monotonic_increasing_scalar_function import (
    PrecomputedMonotonicIncreasingScalarFunction,
)
from elasticai.creator.vhdl.translatable import Translatable


class FPPrecomputedMonotonicIncreasingModule(torch.nn.Module, Translatable):
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
        self._arithmetics = FixedPointArithmetics(self._config)
        self._step_lut = torch.linspace(*sampling_intervall, num_steps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self._stepped_inputs(inputs)
        outputs = self._base_module(inputs)
        return self._arithmetics.quantize(outputs)

    def translate(self, name: str) -> Design:
        quantized_inputs = list(map(self._config.as_integer, self._step_lut.tolist()))
        return PrecomputedMonotonicIncreasingScalarFunction(
            name=name,
            width=self._config.total_bits,
            inputs=quantized_inputs,
            function=self._quantized_inference,
        )

    def _stepped_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        step_inputs = cast(
            torch.Tensor, IdentityStepFunction.apply(inputs, self._step_lut)
        )
        return self._arithmetics.quantize(step_inputs)

    def _quantized_inference(self, x: int) -> int:
        fp_input = self._config.as_rational(x)
        with torch.no_grad():
            output = self(torch.tensor(fp_input))
        return self._config.as_integer(float(output.item()))
