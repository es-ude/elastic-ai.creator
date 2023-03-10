from itertools import chain, repeat

from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.design_base.network_blocks import BufferedNetworkBlock
from elasticai.creator.hdl.vhdl.code_generation import to_vhdl_hex_string
from elasticai.creator.hdl.vhdl.code_generation.template import Template


class FPLSTMCell(BufferedNetworkBlock):
    def __init__(
        self,
        *,
        total_bits: int,
        frac_bits: int,
        hidden_size: int,
        input_size: int,
    ):
        super().__init__(
            name="lstm",
            x_width=total_bits,
            y_width=total_bits,
            y_count=hidden_size + input_size,
            x_count=hidden_size + input_size,
        )
        self.total_bits = total_bits
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.frac_bits = frac_bits
        self.x_h_addr_width = calculate_address_width(
            self.input_size + self.hidden_size
        )
        self.hidden_addr_width = calculate_address_width(self.hidden_size)
        self.w_addr_width = calculate_address_width(
            (self.input_size + self.hidden_size) * self.hidden_size
        )

    def _save_bf_rom(self):
        ...

    def _save_bg_rom(self):
        ...

    def _save_bi_rom(self):
        ...

    def _save_bo_rom(self):
        ...

    def _save_wf_rom(self):
        ...

    def _save_wg_rom(self):
        ...

    def _save_wi_rom(self):
        ...

    def _save_wo_rom(self):
        ...

    def lines(self) -> list[str]:
        template = Template(base_name="lstm")
        template.update_parameters(
            work_library_name="work",
            data_width=str(self.total_bits),
            frac_width=str(self.frac_bits),
            input_size=str(self.input_size),
            hidden_size=str(self.hidden_size),
            x_h_addr_width=str(self.x_h_addr_width),
            hidden_addr_width=str(self.hidden_addr_width),
            w_addr_width=str(self.w_addr_width),
        )
        return template.lines()


class _DualPortDoubleClockRom:
    def __init__(
        self,
        data_width: int,
        values: list[int],
        name: str,
        resource_option: str,
    ) -> None:
        self.name = name
        self.resource_option = resource_option
        self.data_width = data_width
        self.addr_width = calculate_address_width(len(values))
        padded_values = chain(values, repeat(0, 2**self.addr_width))

        def to_hex(number: int) -> str:
            return to_vhdl_hex_string(number=number, bit_width=self.data_width)

        self.hex_values = list(map(to_hex, padded_values))

    def lines(self) -> list[str]:
        template = Template(base_name="rom")
        template.update_parameters(
            name=self.name,
            rom_addr_bitwidth=str(self.addr_width),
            rom_data_bitwidth=str(self.data_width),
            rom_value=",".join(self.hex_values),
            rom_resource_option=f'"{self.resource_option}"',
        )

        return template.lines()
