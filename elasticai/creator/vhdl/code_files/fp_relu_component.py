from dataclasses import dataclass
from typing import Callable

from elasticai.creator.vhdl.code.code import Code
from elasticai.creator.vhdl.language.vhdl_template import VHDLTemplate
from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    fixed_point_params_from_factory,
)


@dataclass
class FPReLUComponent:
    layer_id: str  # used to distinguish layers in the same model
    fixed_point_factory: Callable[[float], FixedPoint]

    def __post_init__(self) -> None:
        self.data_width, self.frac_width = fixed_point_params_from_factory(
            self.fixed_point_factory
        )

    @property
    def file_name(self) -> str:
        return f"fp_relu_{self.layer_id}.vhd"

    def code(self) -> Code:
        template = VHDLTemplate(template_name="fp_relu")
        template.update_parameters(
            layer_name=self.layer_id,
            data_width=str(self.data_width),
            clock_option="false",
        )

        return template.lines()
