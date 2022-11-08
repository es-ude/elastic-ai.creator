from dataclasses import dataclass
from typing import Callable

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    fixed_point_params_from_factory,
)


@dataclass
class FPHardSigmoidComponent:
    zero_threshold: FixedPoint
    one_threshold: FixedPoint
    slope: FixedPoint
    y_intercept: FixedPoint
    fixed_point_factory: Callable[[float], FixedPoint]

    def __post_init__(self) -> None:
        self.data_width, self.frac_width = fixed_point_params_from_factory(
            self.fixed_point_factory
        )

    @property
    def file_name(self) -> str:
        return "fp_hard_sigmoid.vhd"

    def __call__(self) -> Code:
        template = read_text(
            "elasticai.creator.vhdl.templates", "fp_hard_sigmoid.tpl.vhd"
        )

        code = template.format(
            data_width=self.data_width,
            frac_width=self.frac_width,
            one=self.fixed_point_factory(1).to_signed_int(),
            zero_threshold=self.zero_threshold.to_signed_int(),
            one_threshold=self.one_threshold.to_signed_int(),
            slope=self.slope.to_signed_int(),
            y_intercept=self.y_intercept.to_signed_int(),
        )

        yield from code.splitlines()
