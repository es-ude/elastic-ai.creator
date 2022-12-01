from dataclasses import dataclass
from typing import Callable, Iterable

from elasticai.creator.vhdl.components.fp_relu_component import FPReLUComponent
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.vhdl_files import VHDLFile, VHDLModule


@dataclass
class FPReLUTranslationArgs:
    fixed_point_factory: Callable[[float], FixedPoint]


@dataclass
class FPReLUModule(VHDLModule):
    layer_id: str

    @property
    def name(self) -> str:
        return self.layer_id

    def files(self, args: FPReLUTranslationArgs) -> Iterable[VHDLFile]:
        yield FPReLUComponent(
            layer_id=self.layer_id,
            fixed_point_factory=args.fixed_point_factory,
        )
