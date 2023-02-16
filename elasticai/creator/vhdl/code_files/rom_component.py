from collections.abc import Sequence
from dataclasses import dataclass

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.code import Code, CodeFileBase
from elasticai.creator.vhdl.code_files.utils import (
    calculate_address_width,
    pad_with_zeros,
)
from elasticai.creator.vhdl.language import hex_representation
from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    infer_total_and_frac_bits,
)
from elasticai.creator.vhdl.vhdl_files import expand_template


@dataclass
class RomFile(CodeFileBase):
    rom_name: str
    layer_id: str
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
        return f"{self.rom_name}_{self.layer_id}.vhd"

    def code(self) -> Code:
        template = read_text("elasticai.creator.vhdl.templates", "rom.tpl.vhd")

        code = expand_template(
            template,
            rom_name=self.rom_name,
            layer_name=self.layer_id,
            rom_addr_bitwidth=str(self.addr_width),
            rom_data_bitwidth=str(self.data_width),
            rom_value=",".join(self.hex_values),
            rom_resource_option=f'"{self.resource_option}"',
        )

        return code
