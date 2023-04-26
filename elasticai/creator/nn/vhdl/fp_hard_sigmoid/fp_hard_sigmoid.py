from abc import abstractmethod
from typing import Protocol

from elasticai.creator.base_modules.hard_sigmoid import HardSigmoid as _HardSigmoidLayer
from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.translatable import Translatable
from elasticai.creator.nn.two_complement_fixed_point_config import FixedPointConfig
from elasticai.creator.nn.vhdl.fp_hard_sigmoid.design import HardSigmoid


class FixedPointOps(Protocol):
    @property
    @abstractmethod
    def total_bits(self) -> int:
        ...

    @property
    @abstractmethod
    def frac_bits(self) -> int:
        ...


class FPHardSigmoid(_HardSigmoidLayer, Translatable):
    def __init__(self, ops: FixedPointOps):
        super().__init__()
        self.ops = ops

    def translate(self, name: str) -> Design:
        conf = FixedPointConfig(
            total_bits=self.ops.total_bits, frac_bits=self.ops.frac_bits
        )

        return HardSigmoid(
            lower_bound_for_zero=conf.as_integer(-3),
            upper_bound_for_one=conf.as_integer(3),
            width=self.ops.total_bits,
        )
