from collections.abc import Sequence
from dataclasses import dataclass

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.components.utils import (
    calculate_address_width,
    pad_with_zeros,
)
from elasticai.creator.vhdl.language import hex_representation
from vhdl.code import Code
from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    infer_total_and_frac_bits,
)
from elasticai.creator.vhdl.templates.utils import expand_template
from elasticai.creator.vhdl.vhdl_files import VHDLFile


@dataclass
class RomComponent(VHDLFile):
    rom_name: str
    values: Sequence[FixedPoint]
    resource_option: str

    def __post_init__(self) -> None:
        self.data_width, _ = infer_total_and_frac_bits(self.values)
        self.addr_width = calculate_address_width(len(self.values))
        padded_values = pad_with_zeros(list(self.values), 2**self.addr_width)
        self.hex_values = list(
            map(lambda fp: hex_representation(fp.to_hex()), padded_values)
        )

    @property
    def name(self) -> str:
        return f"{self.rom_name}.vhd"

    @property
    def code(self) -> Code:
        template = read_text("elasticai.creator.vhdl.templates", "rom.tpl.vhd")

        code = expand_template(
            template,
            rom_name=self.rom_name,
            rom_addr_bitwidth=self.addr_width,
            rom_data_bitwidth=self.data_width,
            rom_value=",".join(self.hex_values),
            rom_resource_option=f'"{self.resource_option}"',
        )

        yield from code
