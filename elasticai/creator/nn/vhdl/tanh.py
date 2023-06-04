from typing import cast

import torch

from elasticai.creator.base_modules.tanh import Tanh
from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.translatable import Translatable
from elasticai.creator.hdl.vhdl.designs.monotonously_increasing_precomputed_scalar_function.precomputed_scalar_function import (
    PrecomputedMonotonouslyIncreasingScalarFunction,
)
from elasticai.creator.nn.fixed_point_arithmetics import FixedPointArithmetics
from elasticai.creator.nn.two_complement_fixed_point_config import FixedPointConfig


class FPTanh(Tanh, Translatable):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-5, 5),
    ) -> None:
        super().__init__(
            arithmetics=FixedPointArithmetics(
                config=FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
            ),
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
        self._config = cast(FixedPointArithmetics, self._arithmetics).config

    def _quantized_inference(self, x: int) -> int:
        fp_input = self._config.as_rational(x)
        with torch.no_grad():
            output = self(torch.tensor(fp_input))
        return self._config.as_integer(float(output.item()))

    def translate(self, name: str) -> Design:
        quantized_inputs = list(map(self._config.as_integer, self._step_lut.tolist()))
        return PrecomputedMonotonouslyIncreasingScalarFunction(
            name=name,
            width=self._config.total_bits,
            inputs=quantized_inputs,
            function=self._quantized_inference,
        )
