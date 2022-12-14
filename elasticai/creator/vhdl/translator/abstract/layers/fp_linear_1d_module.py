from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import chain
from typing import Callable, Collection

import numpy as np

from elasticai.creator.vhdl.code import CodeFile, CodeModule
from elasticai.creator.vhdl.code_files.fp_linear_1d_component import FPLinear1dFile
from elasticai.creator.vhdl.code_files.rom_component import RomFile
from elasticai.creator.vhdl.number_representations import FixedPoint


@dataclass
class FPLinear1dTranslationArgs:
    fixed_point_factory: Callable[[float], FixedPoint]
    work_library_name: str = field(default="work")


@dataclass
class FPLinear1dModule(CodeModule):
    @property
    def submodules(self) -> Collection["CodeModule"]:
        raise NotImplementedError()

    layer_id: str
    weight: list[list[float]]
    bias: list[float]

    @property
    def name(self) -> str:
        return self.layer_id

    def files(self, args: FPLinear1dTranslationArgs) -> Iterator[CodeFile]:
        def to_fp(values: Iterable[float]) -> list[FixedPoint]:
            return list(map(args.fixed_point_factory, values))

        out_features, in_features = np.shape(self.weight)

        yield FPLinear1dFile(
            layer_id=self.layer_id,
            in_feature_num=in_features,
            out_feature_num=out_features,
            fixed_point_factory=args.fixed_point_factory,
            work_library_name=args.work_library_name,
            resource_option="auto",
        )

        flat_weight = chain(*self.weight)

        name_suffix = f"_fp_linear_1d_{self.layer_id}"
        yield RomFile(
            rom_name="w_rom" + name_suffix,
            values=to_fp(flat_weight),
            resource_option="auto",
        )

        yield RomFile(
            rom_name="b_rom" + name_suffix,
            values=to_fp(self.bias),
            resource_option="auto",
        )
