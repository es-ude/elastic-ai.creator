import math
from dataclasses import dataclass, field
from typing import Callable

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import FixedPoint


@dataclass
class LSTMComponent:
    input_size: int
    hidden_size: int
    fixed_point_factory: Callable[[float], FixedPoint]
    work_library_name: str = field(default="work")

    def __post_init__(self):
        self.data_width, self.frac_width = self._derive_data_and_frac_width(
            self.fixed_point_factory
        )
        self.x_h_addr_width = self._calculate_addr_width(
            self.input_size + self.hidden_size
        )
        self.hidden_addr_width = self._calculate_addr_width(self.input_size)
        self.w_addr_width = self._calculate_addr_width(
            (self.input_size + self.hidden_size) * self.hidden_size
        )

    @staticmethod
    def _derive_data_and_frac_width(
        fixed_point_factory: Callable[[float], FixedPoint]
    ) -> tuple[int, int]:
        value = fixed_point_factory(0)
        return value.total_bits, value.frac_bits

    @staticmethod
    def _calculate_addr_width(num_items: int) -> int:
        return max(1, math.ceil(math.log2(num_items)))

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
