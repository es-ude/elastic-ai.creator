from dataclasses import dataclass
from typing import Collection

from elasticai.creator.vhdl.code_files.fp_relu_component import FPReLUComponent
from elasticai.creator.vhdl.number_representations import FixedPointConfig


@dataclass
class FPReLUModule:
    layer_id: str
    fixed_point_factory: FixedPointConfig

    @property
    def submodules(self) -> Collection:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.layer_id

    @property
    def files(self) -> Collection:
        yield FPReLUComponent(
            layer_id=self.layer_id,
            fixed_point_factory=self.fixed_point_factory,
        )
