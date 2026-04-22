from typing import cast

import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.math_operations import MathOperations
from elasticai.creator.vhdl.shared_designs.precomputed_scalar_function import (
    PrecomputedScalarFunction,
)

from .identity_step_function import IdentityStepFunction


class PrecomputedModule(DesignCreatorModule):
    _xoffset: float
    _lut_input: torch.nn.Buffer

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
        self._params = FxpParams(
            total_bits=total_bits, frac_bits=frac_bits, signed=True
        )
        self._config = FxpArithmetic(self._params)
        self._operations = MathOperations(self._config)
        self._build_lut(in_range=sampling_intervall, lut_size=num_steps)

    def _build_lut(self, in_range: tuple[float, float], lut_size: int) -> None:
        range_neg = (
            self._config.minimum_as_rational
            if abs(in_range[0]) == float("inf")
            else in_range[0]
        )
        range_pos = (
            self._config.maximum_as_rational
            if abs(in_range[1]) == float("inf")
            else in_range[1]
        )
        lut_num_steps = (
            2**self._config.total_bits
            if lut_size > 2**self._config.total_bits
            else lut_size
        )

        self._lut_input = torch.nn.Buffer(
            self._operations.round(
                torch.linspace(start=range_neg, end=range_pos, steps=lut_num_steps)
            ),
            persistent=False,
        )
        lut_diff = (
            torch.abs(torch.diff(self._lut_input))
            / self._params.minimum_step_as_rational
            / 2
        )
        self._xoffset = (
            float((lut_diff.max() + lut_diff.min()) / 2)
            * self._params.minimum_step_as_rational
        )

    def get_lut_integer(self) -> tuple[list[int], list[int]]:
        return (
            list(map(self._config.cut_as_integer, self._lut_input.tolist())),
            [self._forward_nograd(val) for val in self._lut_input.tolist()],
        )

    def _stepped_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, IdentityStepFunction.apply(x, self._lut_input))

    def _forward_nograd(self, x: int) -> int:
        fxp_input = self._config.as_rational(x)
        if not isinstance(fxp_input, float):
            raise ValueError()

        with torch.no_grad():
            output = self.forward(torch.tensor(fxp_input).clone().detach())
        return self._config.round_to_integer(float(output.item()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._stepped_inputs(x)
        y = self._base_module(x - self._xoffset)
        return self._operations.round(y)

    def create_design(self, name: str) -> PrecomputedScalarFunction:
        q_input = self.get_lut_integer()[0]
        return PrecomputedScalarFunction(
            name=name,
            input_width=self._config.total_bits,
            output_width=self._config.total_bits,
            inputs=q_input,
            function=self._forward_nograd,
        )
