from dataclasses import dataclass, field
from typing import Callable

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.components.utils import (
    calculate_addr_width,
    derive_fixed_point_params_from_factory,
)
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import FixedPoint


@dataclass
class Linear1dComponent:
    in_features: int
    out_features: int
    fixed_point_factory: Callable[[float], FixedPoint]
    work_library_name: str = field(default="work")

    def __post_init__(self) -> None:
        if self.out_features != 1:
            raise NotImplementedError(
                "Currently only one bias is supported (which implies that out_features must be 1)."
            )

        self.data_width, _ = derive_fixed_point_params_from_factory(
            self.fixed_point_factory
        )
        self.addr_width = calculate_addr_width(self.in_features)

    @property
    def file_name(self) -> str:
        return "linear_1d.vhd"

    def __call__(self) -> Code:
        template = read_text("elasticai.creator.vhdl.templates", "linear_1d.tpl.vhd")

        code = template.format(
            work_library_name=self.work_library_name,
            addr_width=self.addr_width,
            data_width=self.out_features,
            in_feature_count=self.in_features,
            out_feature_count=self.out_features,
        )

        yield from code.splitlines()
