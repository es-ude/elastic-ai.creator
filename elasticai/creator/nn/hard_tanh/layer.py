from elasticai.creator.base_modules.hard_tanh import HardTanh
from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.vhdl.design_base.design import Design
from elasticai.creator.vhdl.translatable import Translatable

from .design import FPHardTanh as FPHardTanhDesign


class FPHardTanh(Translatable, HardTanh):
    def __init__(
        self, total_bits: int, frac_bits: int, min_val: float = -1, max_val: float = 1
    ) -> None:
        super().__init__(min_val, max_val)
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)

    def translate(self, name: str) -> Design:
        return FPHardTanhDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            min_val=self._config.as_integer(self.min_val),
            max_val=self._config.as_integer(self.max_val),
        )
