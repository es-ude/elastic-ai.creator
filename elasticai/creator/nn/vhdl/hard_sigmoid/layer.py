from elasticai.creator.base_modules.hard_sigmoid import HardSigmoid
from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.translatable import Translatable
from elasticai.creator.nn.fixed_point_arithmetics import FixedPointArithmetics
from elasticai.creator.nn.two_complement_fixed_point_config import FixedPointConfig

from .design import FPHardSigmoid as FPHardSigmoidDesign


class FPHardSigmoid(Translatable, HardSigmoid):
    def __init__(self, total_bits: int, frac_bits: int) -> None:
        super().__init__()
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)

    def translate(self, name: str) -> Design:
        return FPHardSigmoidDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            one=self._config.as_integer(1),
            zero_threshold=self._config.as_integer(-3),
            one_threshold=self._config.as_integer(3),
            slope=self._config.as_integer(1 / 6),
            y_intercept=self._config.as_integer(0.5),
        )
