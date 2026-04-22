from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.base_modules.hard_sigmoid import HardSigmoid as HardSigmoidBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule

from .design import HardSigmoid as HardSigmoidDesign


class HardSigmoid(DesignCreatorModule, HardSigmoidBase):
    def __init__(self, total_bits: int, frac_bits: int) -> None:
        """Quantized Activation Function for Sigmoid
        :param total_bits:          Total number of bits
        :param frac_bits:           Fraction of bits
        """
        super().__init__()
        self._params = FxpParams(
            total_bits=total_bits, frac_bits=frac_bits, signed=True
        )
        self._config = FxpArithmetic(self._params)

    def create_design(self, name: str) -> HardSigmoidDesign:
        neg_thr = -3.0
        if self._params.rational_out_of_bounds(neg_thr):
            neg_thr = self._config.minimum_as_rational
        pos_thr = +3.0
        if self._params.rational_out_of_bounds(pos_thr):
            pos_thr = self._config.maximum_as_rational
        y_int = 0.5
        if self._params.rational_out_of_bounds(y_int):
            pos_thr = self._config.maximum_as_rational
        y_max = 1.0
        if self._params.rational_out_of_bounds(y_max):
            y_max = self._config.maximum_as_rational
        slope = 1 / 6
        if self._params.rational_out_of_bounds(slope):
            y_max = self._params.minimum_step_as_rational

        return HardSigmoidDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            one=self._config.cut_as_integer(y_max),
            zero_threshold=self._config.cut_as_integer(neg_thr),
            one_threshold=self._config.cut_as_integer(pos_thr),
            slope=self._config.cut_as_integer(slope),
            y_intercept=self._config.cut_as_integer(y_int),
        )
