from dataclasses import dataclass
from typing import Callable, Collection, Iterable

from elasticai.creator.vhdl.code import CodeFile, CodeModule
from elasticai.creator.vhdl.code_files.fp_relu_component import FPReLUComponent
from elasticai.creator.vhdl.number_representations import FixedPoint


@dataclass
class FPReLUTranslationArgs:
    fixed_point_factory: Callable[[float], FixedPoint]


@dataclass
class FPReLUModule(CodeModule):
    @property
    def submodules(self) -> Collection["CodeModule"]:
        raise NotImplementedError()

    layer_id: str

    @property
    def name(self) -> str:
        return self.layer_id

    def files(self, args: FPReLUTranslationArgs) -> Iterable[CodeFile]:
        yield FPReLUComponent(
            layer_id=self.layer_id,
            fixed_point_factory=args.fixed_point_factory,
        )
