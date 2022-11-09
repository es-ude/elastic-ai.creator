from dataclasses import dataclass
from typing import Callable, Iterable

from elasticai.creator.vhdl.components.fp_relu_component import FPReLUComponent
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.vhdl_component import VHDLComponent, VHDLModule


@dataclass
class FPReLUTranslationArgs:
    fixed_point_factory: Callable[[float], FixedPoint]


@dataclass
class FPReLUModule(VHDLModule):
    def components(self, args: FPReLUTranslationArgs) -> Iterable[VHDLComponent]:
        yield FPReLUComponent(
            fixed_point_factory=args.fixed_point_factory,
        )
