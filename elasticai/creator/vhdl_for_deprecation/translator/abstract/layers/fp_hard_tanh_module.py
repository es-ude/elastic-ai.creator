from dataclasses import dataclass
from typing import Iterator

from elasticai.creator.hdl.vhdl.code_files import FPHardTanhComponent
from elasticai.creator.vhdl.number_representations import FixedPointConfig
from elasticai.creator.vhdl_for_deprecation.translator.pytorch.code_module_base import (
    CodeModuleBase,
)


@dataclass
class FPHardTanhModule(CodeModuleBase):
    fixed_point_factory: FixedPointConfig
    layer_id: str

    @property
    def name(self) -> str:
        return self.layer_id

    @property
    def files(self) -> Iterator:
        yield FPHardTanhComponent(
            min_val=self.fixed_point_factory(-1),
            max_val=self.fixed_point_factory(1),
            fixed_point_factory=self.fixed_point_factory,
            layer_id=self.layer_id,
        )
