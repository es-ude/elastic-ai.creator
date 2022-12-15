from dataclasses import dataclass
from typing import Callable, Collection, Iterable

from elasticai.creator.vhdl.code import CodeFile, CodeModule
from elasticai.creator.vhdl.code_files.fp_hard_sigmoid_file import FPHardSigmoidFile
from elasticai.creator.vhdl.number_representations import FixedPoint


@dataclass
class FPHardSigmoidTranslationArgs:
    fixed_point_factory: Callable[[float], FixedPoint]


@dataclass
class FPHardSigmoidModule(CodeModule):
    @property
    def submodules(self) -> Collection["CodeModule"]:
        raise NotImplementedError()

    layer_id: str

    @property
    def name(self) -> str:
        self.layer_id

    def files(self, args: FPHardSigmoidTranslationArgs) -> Iterable[CodeFile]:
        yield FPHardSigmoidFile(
            layer_id=self.layer_id,
            zero_threshold=args.fixed_point_factory(-3),
            one_threshold=args.fixed_point_factory(3),
            slope=args.fixed_point_factory(0.125),
            y_intercept=args.fixed_point_factory(0.5),
            fixed_point_factory=args.fixed_point_factory,
        )
