from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import chain
from typing import Callable

import numpy as np

from elasticai.creator.vhdl.components import FPLinear1dComponent, RomComponent
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.vhdl_component import VHDLComponent, VHDLModule


@dataclass
class FPLinear1dTranslationArgs:
    fixed_point_factory: Callable[[float], FixedPoint]
    work_library_name: str = field(default="work")


@dataclass
class FPLinear1dModule(VHDLModule):
    layer_name: str
    weight: list[list[float]]
    bias: list[float]

    def components(self, args: FPLinear1dTranslationArgs) -> Iterator[VHDLComponent]:
        def to_fp(values: Iterable[float]) -> list[FixedPoint]:
            return list(map(args.fixed_point_factory, values))

        out_features, in_features = np.shape(self.weight)

        yield FPLinear1dComponent(
            layer_name=self.layer_name,
            in_features=in_features,
            out_features=out_features,
            fixed_point_factory=args.fixed_point_factory,
            work_library_name=args.work_library_name,
            resource_option="auto",
        )

        flat_weight = chain(*self.weight)

        name_suffix = "_fp_linear_1d_{layer_name}".format(layer_name=self.layer_name)
        yield RomComponent(
            rom_name="w_rom" + name_suffix,
            values=to_fp(flat_weight),
            resource_option="auto",
        )

        yield RomComponent(
            rom_name="b_rom" + name_suffix,
            values=to_fp(self.bias),
            resource_option="auto",
        )
