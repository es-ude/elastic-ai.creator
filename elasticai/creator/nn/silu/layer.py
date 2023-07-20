from elasticai.creator.base_modules.silu import SiLU
from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.vhdl.design_base.design import Design
from elasticai.creator.vhdl.translatable import Translatable

from ...base_modules.arithmetics.arithmetics import Arithmetics
from .design import FPSiLU as FPSiLUDesign


class FPSiLU(Translatable, SiLU):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        arithmetics: Arithmetics,
    ) -> None:
        super().__init__(arithmetics)
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)

    def translate(self, name: str) -> Design:
        return FPSiLUDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
        )
