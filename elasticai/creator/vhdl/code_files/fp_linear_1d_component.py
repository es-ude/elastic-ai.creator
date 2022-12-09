from dataclasses import dataclass, field
from importlib.resources import read_text
from typing import Callable

from vhdl.code_files.utils import (
    calculate_address_width,
    derive_fixed_point_params_from_factory,
)
from vhdl.code import Code, CodeFile
from elasticai.creator.vhdl.number_representations import FixedPoint
from vhdl.hw_equivalent_layers.vhdl_files import expand_template


@dataclass
class FPLinear1dFile(CodeFile):
    def save_to(self, prefix: str):
        raise NotImplementedError()

    layer_id: str  # used to distinguish layers in the same model
    in_features: int
    out_features: int
    fixed_point_factory: Callable[[float], FixedPoint]
    work_library_name: str = field(default="work")
    resource_option: str = "auto"

    def __post_init__(self) -> None:
        self.data_width, self.frac_width = derive_fixed_point_params_from_factory(
            self.fixed_point_factory
        )
        self.x_addr_width = calculate_address_width(self.in_features)
        self.y_addr_width = calculate_address_width(self.out_features)

    @property
    def name(self) -> str:
        return f"fp_linear_1d_{self.layer_id}.vhd"

    @property
    def code(self) -> Code:
        template = read_text("elasticai.creator.vhdl.templates", "fp_linear_1d.tpl.vhd")

        code = expand_template(
            template.splitlines(),
            layer_name=self.layer_id,
            work_library_name=self.work_library_name,
            data_width=self.data_width,
            frac_width=self.frac_width,
            x_addr_width=self.x_addr_width,
            y_addr_width=self.y_addr_width,
            in_feature_num=self.in_features,
            out_feature_num=self.out_features,
            resource_option=f'"{self.resource_option}"',
        )

        yield from code
