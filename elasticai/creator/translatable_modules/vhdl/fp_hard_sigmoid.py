from abc import abstractmethod
from typing import Protocol

from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.vhdl.designs.fp_hard_sigmoid import (
    FPHardSigmoid as _FPHardSigmoidDesign,
)
from elasticai.creator.hdl.vhdl.number_representations import (
    FixedPoint,
    FixedPointConfig,
)
from elasticai.creator.nn.hard_sigmoid import HardSigmoid as _HardSigmoidLayer


class FixedPointOps(Protocol):
    @property
    @abstractmethod
    def total_bits(self) -> int:
        ...

    @property
    @abstractmethod
    def frac_bits(self) -> int:
        ...


class FPHardSigmoid(_HardSigmoidLayer):
    def __init__(self, ops: FixedPointOps):
        super().__init__()
        self.ops = ops

    def translate(self) -> Design:
        def fp(value: float) -> FixedPoint:
            return FixedPoint(
                value=value,
                total_bits=self.ops.total_bits,
                frac_bits=self.ops.frac_bits,
            )

        return _FPHardSigmoidDesign(
            zero_threshold=fp(-3),
            one_threshold=fp(3),
            slope=fp(0.66),
            y_intercept=fp(0.5),
            fixed_point_factory=FixedPointConfig(
                frac_bits=self.ops.frac_bits,
                total_bits=self.ops.total_bits,
                constructor=FixedPoint,
            ),
        )
