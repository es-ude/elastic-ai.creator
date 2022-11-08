from dataclasses import dataclass
from typing import Callable, Iterable

from elasticai.creator.vhdl.components.fp_hard_sigmoid_componet import (
    FPHardSigmoidComponent,
)
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.vhdl_component import VHDLComponent, VHDLModule


@dataclass
class FPHardSigmoidTranslationArgs:
    fixed_point_factory: Callable[[float], FixedPoint]


@dataclass
class FPHardSigmoidModule(VHDLModule):
    def components(self, args: FPHardSigmoidTranslationArgs) -> Iterable[VHDLComponent]:
        yield FPHardSigmoidComponent(
            zero_threshold=args.fixed_point_factory(-3),
            one_threshold=args.fixed_point_factory(3),
            slope=args.fixed_point_factory(0.125),
            y_intercept=args.fixed_point_factory(0.5),
            fixed_point_factory=args.fixed_point_factory,
        )
