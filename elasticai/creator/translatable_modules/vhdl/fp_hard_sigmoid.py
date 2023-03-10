from abc import abstractmethod
from typing import Protocol

from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.vhdl.designs.monotonously_increasing_precomputed_scalar_function.hard_sigmoid import (
    HardSigmoid,
)
from elasticai.creator.hdl.vhdl.number_representations import FixedPoint
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
        def fp(value: float) -> int:
            return int(
                FixedPoint(
                    value=value,
                    total_bits=self.ops.total_bits,
                    frac_bits=self.ops.frac_bits,
                )
            )

        return HardSigmoid(
            lower_bound_for_zero=fp(-3),
            upper_bound_for_one=fp(3),
            width=self.ops.total_bits,
        )
