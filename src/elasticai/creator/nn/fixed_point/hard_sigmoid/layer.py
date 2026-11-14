from dataclasses import dataclass

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.base_modules.hard_sigmoid import HardSigmoid as HardSigmoidBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.math_operations import MathOperations

from .design import HardSigmoid as HardSigmoidDesign


@dataclass(frozen=True)
class ParamsHardSigmoid:
    bias: int
    slope: int
    xmin: int
    xmax: int
    ymin: int
    ymax: int


class HardSigmoid(DesignCreatorModule, HardSigmoidBase):
    def __init__(self, total_bits: int, frac_bits: int, slope: float = 0.25) -> None:
        """Quantized Activation Function for Sigmoid
        :param total_bits:          Total number of bits
        :param frac_bits:           Fraction of bits
        :param slope:               Slope
        """
        self._params = FxpParams(
            total_bits=total_bits, frac_bits=frac_bits, signed=True
        )
        self._config = FxpArithmetic(self._params)
        super().__init__(mathops=MathOperations(self._config), slope=slope)

    def get_ranges(self) -> ParamsHardSigmoid:
        xmin, xmax = (-0.5 / self._slope, +0.5 / self._slope)
        ymin, ymax = (0.0, +1.0)
        bias = 0.5

        if self._params.rational_out_underflow(xmin):
            xmin = self._params.minimum_as_integer
        else:
            xmin = self._config.cut_as_integer(xmin)

        if self._params.rational_out_overflow(xmax):
            xmax = self._params.maximum_as_integer
        else:
            xmax = self._config.cut_as_integer(xmax)

        ymin = 0
        if self._params.rational_out_overflow(ymax):
            ymax = self._params.maximum_as_integer
        else:
            ymax = self._config.cut_as_integer(ymax)

        if self._params.rational_out_overflow(bias):
            bias = self._params.maximum_as_integer
        else:
            bias = self._config.cut_as_integer(bias)

        if self._params.rational_out_overflow(self._slope):
            slope = self._params.maximum_as_integer
        else:
            slope = self._config.cut_as_integer(self._slope)
        return ParamsHardSigmoid(
            bias=bias, slope=slope, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
        )

    def create_design(self, name: str) -> HardSigmoidDesign:
        params = self.get_ranges()

        return HardSigmoidDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            one=params.ymax,
            zero_threshold=params.xmin,
            one_threshold=params.xmax,
            slope=params.slope,
            y_intercept=params.bias,
        )
