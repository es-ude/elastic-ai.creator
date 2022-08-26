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
class LSTMComponent:
    input_size: int
    hidden_size: int
    fixed_point_factory: Callable[[float], FixedPoint]
    work_library_name: str = field(default="work")

    def __post_init__(self) -> None:
        self.data_width, self.frac_width = derive_fixed_point_params_from_factory(
            self.fixed_point_factory
        )
        self.x_h_addr_width = calculate_addr_width(self.input_size + self.hidden_size)
        self.hidden_addr_width = calculate_addr_width(self.input_size)
        self.w_addr_width = calculate_addr_width(
            (self.input_size + self.hidden_size) * self.hidden_size
        )

    @property
    def file_name(self) -> str:
        return "lstm.vhd"

    def __call__(self) -> Code:
        template = read_text("elasticai.creator.vhdl.templates", "lstm.tpl.vhd")

        code = template.format(
            work_library_name=self.work_library_name,
            data_width=self.data_width,
            frac_width=self.frac_width,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            x_h_addr_width=self.x_h_addr_width,
            hidden_addr_width=self.hidden_addr_width,
            w_addr_width=self.w_addr_width,
        )

        yield from code.splitlines()
