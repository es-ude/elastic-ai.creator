from dataclasses import dataclass, field
from typing import Callable

from elasticai.creator.hdl.vhdl.code_files import (
    calculate_address_width,
    derive_fixed_point_params_from_factory,
)
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.templates import VHDLTemplate


@dataclass
class FPLinear1dFile:
    def save_to(self, prefix: str):
        raise NotImplementedError()

    layer_id: str  # used to distinguish layers in the same model
    in_feature_num: int
    out_feature_num: int
    fixed_point_factory: Callable[[float], FixedPoint]
    work_library_name: str = field(default="work")
    resource_option: str = "auto"

    def __post_init__(self) -> None:
        self.data_width, self.frac_width = derive_fixed_point_params_from_factory(
            self.fixed_point_factory
        )
        self.x_addr_width = calculate_address_width(self.in_feature_num)
        self.y_addr_width = calculate_address_width(self.out_feature_num)

    def _template_parameters(self) -> dict[str, str]:
        return dict(
            (key, str(getattr(self, key)))
            for key in (
                "data_width",
                "frac_width",
                "x_addr_width",
                "y_addr_width",
                "in_feature_num",
                "out_feature_num",
            )
        )

    @property
    def name(self) -> str:
        return f"fp_linear_1d_{self.layer_id}.vhd"

    def code(self) -> list[str]:
        template = VHDLTemplate(base_name="fp_linear_1d")
        template.update_parameters(
            layer_name=self.layer_id,
            work_library_name=self.work_library_name,
            resource_option=f'"{self.resource_option}"',
            **self._template_parameters(),
        )
        return template.lines()
