from elasticai.creator.base_modules.hard_sigmoid import HardSigmoid as HardSigmoidBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.two_complement_fixed_point_config import (
    FixedPointConfig,
)

from .design import HardSigmoid as HardSigmoidDesign


class HardSigmoid(DesignCreatorModule, HardSigmoidBase):
    def __init__(self, total_bits: int, frac_bits: int) -> None:
        """Quantized Activation Function for Sigmoid
        :param total_bits:          Total number of bits
        :param frac_bits:           Fraction of bits
        """
        super().__init__()
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)

    def create_design(self, name: str) -> HardSigmoidDesign:
        return HardSigmoidDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            one=self._config.cut_as_integer(1),
            zero_threshold=self._config.cut_as_integer(-3),
            one_threshold=self._config.cut_as_integer(3),
            slope=self._config.cut_as_integer(1 / 6),
            y_intercept=self._config.cut_as_integer(0.5),
        )
