from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import chain
from typing import Iterator

import numpy as np

from elasticai.creator.hdl.vhdl.code_files import FPLinear1dFile, RomFile
from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointConfig


@dataclass
class FPLinear1dModule:
    layer_id: str
    weight: list[list[float]]
    bias: list[float]
    fixed_point_factory: FixedPointConfig
    work_library_name: str = field(default="work")

    @property
    def name(self) -> str:
        return self.layer_id

    @property
    def files(self) -> Iterator:
        def to_fp(values: Iterable[float]) -> list[FixedPoint]:
            return list(map(self.fixed_point_factory, values))

        out_features, in_features = np.shape(self.weight)

        yield FPLinear1dFile(
            layer_id=self.layer_id,
            in_feature_num=in_features,
            out_feature_num=out_features,
            fixed_point_factory=self.fixed_point_factory,
            work_library_name=self.work_library_name,
            resource_option="auto",
        )

        flat_weight = chain(*self.weight)

        name_suffix = "_fp_linear_1d"
        yield RomFile(
            rom_name="w_rom" + name_suffix,
            layer_id=self.layer_id,
            values=to_fp(flat_weight),
            resource_option="auto",
        )

        yield RomFile(
            rom_name="b_rom" + name_suffix,
            layer_id=self.layer_id,
            values=to_fp(self.bias),
            resource_option="auto",
        )
