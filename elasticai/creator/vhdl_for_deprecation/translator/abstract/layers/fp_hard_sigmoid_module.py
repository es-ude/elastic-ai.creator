from collections.abc import Collection
from dataclasses import dataclass
from typing import Iterator

from elasticai.creator.vhdl.code import CodeFile
from elasticai.creator.vhdl.code_files.fp_hard_sigmoid_component import (
    FPHardSigmoidComponent,
)


@dataclass
class FPHardSigmoidModule:
    layer_id: str
    fixed_point_factory: FixedPointConfig

    @property
    def name(self) -> str:
        return self.layer_id

    def files(self) -> Iterator[CodeFile]:
        yield FPHardSigmoidComponent(
            layer_id=self.layer_id,
            zero_threshold=self.fixed_point_factory(-3),
            one_threshold=self.fixed_point_factory(3),
            slope=self.fixed_point_factory(0.125),
            y_intercept=self.fixed_point_factory(0.5),
            fixed_point_factory=self.fixed_point_factory,
        )
