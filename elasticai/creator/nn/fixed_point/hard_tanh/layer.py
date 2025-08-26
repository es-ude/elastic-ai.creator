from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.base_modules.hard_tanh import HardTanh as HardTanhBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule

from .design import HardTanh as HardTanhDesign


class HardTanh(DesignCreatorModule, HardTanhBase):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        min_val: float = -1.0,
        max_val: float = 1.0,
    ) -> None:
        """Quantized Activation Function for Sigmoid
        :param total_bits:   Total number of bits
        :param frac_bits:    Fraction of bits
        :param min_val:      Floating value of the minimal input/output value (downer limitation)
        :param max_val:      Floating value of the minimal input/output value (upper limitation)
        """
        super().__init__(min_val, max_val)
        self._params = FxpParams(
            total_bits=total_bits, frac_bits=frac_bits, signed=True
        )
        self._config = FxpArithmetic(self._params)

        if self._params.rational_out_of_bounds(min_val):
            self.min_val = self._config.minimum_as_rational
        if self._params.rational_out_of_bounds(max_val):
            self.max_val = self._config.maximum_as_rational

    def create_design(self, name: str) -> HardTanhDesign:
        return HardTanhDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            min_val=self._config.cut_as_integer(self.min_val),
            max_val=self._config.cut_as_integer(self.max_val),
        )
