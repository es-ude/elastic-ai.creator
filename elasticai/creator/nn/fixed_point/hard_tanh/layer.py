from elasticai.creator.base_modules.hard_tanh import HardTanh as HardTanhBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfig,
)

from .design import HardTanh as HardTanhDesign


class HardTanh(DesignCreatorModule, HardTanhBase):
    def __init__(
        self, total_bits: int, frac_bits: int, min_val: float = -1, max_val: float = 1
    ) -> None:
        super().__init__(min_val, max_val)
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)

    def create_design(self, name: str) -> HardTanhDesign:
        return HardTanhDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            min_val=self._config.as_integer(self.min_val),
            max_val=self._config.as_integer(self.max_val),
        )
