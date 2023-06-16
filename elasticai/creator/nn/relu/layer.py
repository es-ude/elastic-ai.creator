from elasticai.creator.base_modules.relu import ReLU
from elasticai.creator.vhdl.design_base.design import Design
from elasticai.creator.vhdl.translatable import Translatable

from .design import FPReLU as FPReLUDesign


class FPReLU(Translatable, ReLU):
    def __init__(self, total_bits: int, use_clock: bool = False) -> None:
        super().__init__()
        self._total_bits = total_bits
        self._use_clock = use_clock

    def translate(self, name: str) -> Design:
        return FPReLUDesign(
            name=name,
            data_width=self._total_bits,
            use_clock=self._use_clock,
        )
