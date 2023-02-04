from collections.abc import Collection
from dataclasses import dataclass

from elasticai.creator.vhdl.code import CodeFile, CodeModuleBase
from elasticai.creator.vhdl.code_files.fp_hard_tanh_component import FPHardTanhComponent
from elasticai.creator.vhdl.number_representations import FixedPointFactory


@dataclass
class FPHardTanhModule(CodeModuleBase):
    fixed_point_factory: FixedPointFactory
    layer_id: str

    @property
    def name(self) -> str:
        return self.layer_id

    @property
    def files(self) -> Collection[CodeFile]:
        yield FPHardTanhComponent(
            min_val=self.fixed_point_factory(-1),
            max_val=self.fixed_point_factory(1),
            fixed_point_factory=self.fixed_point_factory,
            layer_id=self.layer_id,
        )
