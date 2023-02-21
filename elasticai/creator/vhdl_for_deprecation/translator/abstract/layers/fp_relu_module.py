from dataclasses import dataclass
from typing import Collection

from elasticai.creator.vhdl.code.code_file import CodeFile
from elasticai.creator.vhdl.code.code_module import CodeModule
from elasticai.creator.vhdl.code_files.fp_relu_component import FPReLUComponent
from elasticai.creator.vhdl.number_representations import FixedPointConfig


@dataclass
class FPReLUModule(CodeModule):
    layer_id: str
    fixed_point_factory: FixedPointConfig

    @property
    def submodules(self) -> Collection["CodeModule"]:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.layer_id

    @property
    def files(self) -> Collection[CodeFile]:
        yield FPReLUComponent(
            layer_id=self.layer_id,
            fixed_point_factory=self.fixed_point_factory,
        )
