from typing import cast

import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.math_operations import MathOperations
from elasticai.creator.vhdl.shared_designs.precomputed_lut_function import (
    PrecomputedLutFunction,
)
from elasticai.creator.vhdl.shared_designs.precomputed_scalar_function import (
    PrecomputedScalarFunction,
)

from .identity_step_function import IdentityStepFunction


class PrecomputedModule(DesignCreatorModule):
    _xoffset: float
    _yoffset: float
    _lut_input: torch.nn.Buffer

    def __init__(
        self,
        base_module: torch.nn.Module,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float],
        use_lut_design: bool = False,
    ) -> None:
        super().__init__()
        self._base_module = base_module
        self._params = FxpParams(
            total_bits=total_bits, frac_bits=frac_bits, signed=True
        )
        self._config = FxpArithmetic(self._params)
        self._operations = MathOperations(self._config)
        self._use_lut_design = use_lut_design
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

        if self._use_lut_design:
            exp_num = int(2**self._config.total_bits / lut_num_steps)
            self._lut_input = torch.nn.Buffer(
                torch.Tensor(
                    [
                        range_neg
                        + val * exp_num * self._config.config.minimum_step_as_rational
                        for val in range(lut_num_steps)
                    ]
                ),
                persistent=False,
            )
            self._xoffset = -self._config.config.minimum_step_as_rational
            self._yoffset = (
                int(exp_num / 2) * self._config.config.minimum_step_as_rational
            )
        else:
            self._lut_input = torch.nn.Buffer(
                self._operations.round(
                    torch.linspace(start=range_neg, end=range_pos, steps=lut_num_steps)
                ),
                persistent=False,
            )
            lut_diff = torch.abs(torch.diff(self._lut_input))
            self._xoffset = 0
            self._yoffset = float((lut_diff.max() + lut_diff.min()) / 4)

    def get_reference_signal(self) -> tuple[list[float], list[float]]:
        x = torch.linspace(
            start=self._config.minimum_as_rational,
            end=self._config.maximum_as_rational,
            steps=4 * len(self._lut_input),
        )
        y = self._base_module(x)
        return (
            (x / self._config.config.minimum_step_as_rational).tolist(),
            (y / self._config.config.minimum_step_as_rational).tolist(),
        )

    def get_lut_integer(self) -> tuple[list[int], list[int]]:
        x = list(map(self._config.cut_as_integer, self._lut_input.tolist()))
        y = [self._forward_nograd(val) for val in self._lut_input.tolist()]
        return (x, y)

    def _stepped_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, IdentityStepFunction.apply(x, self._lut_input))

    def _forward_nograd(self, x: int | float) -> int:
        if isinstance(x, float):
            fxp_input = x
        else:
            fxp_input = self._config.as_rational(x)

        with torch.no_grad():
            output = self.forward(torch.tensor(fxp_input).clone().detach())
        return self._config.cut_as_integer(float(output.item()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._stepped_inputs(x - self._xoffset)
        y = self._base_module(y - self._yoffset)
        return self._operations._round(y)

    def create_design(
        self, name: str
    ) -> PrecomputedScalarFunction | PrecomputedLutFunction:
        q_input = self.get_lut_integer()[0]
        if self._use_lut_design:
            return PrecomputedLutFunction(
                name=name,
                input_width=self._config.total_bits,
                output_width=self._config.total_bits,
                inputs=q_input,
                function=self._forward_nograd,
            )
        else:
            return PrecomputedScalarFunction(
                name=name,
                input_width=self._config.total_bits,
                output_width=self._config.total_bits,
                inputs=q_input,
                function=self._forward_nograd,
            )
