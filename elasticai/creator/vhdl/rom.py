import math
from collections.abc import Sequence

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.language import Code, hex_representation
from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    infer_total_and_frac_bits,
)


def pad_with_zeros(numbers: list[FixedPoint], target_length: int) -> list[FixedPoint]:
    zero = FixedPoint(0, *infer_total_and_frac_bits(numbers))
    return numbers + [zero] * (target_length - len(numbers))


class Rom:
    def __init__(
        self,
        rom_name: str,
        values: Sequence[FixedPoint],
        resource_option: str,
    ):
        self.rom_name = rom_name
        self.data_width, _ = infer_total_and_frac_bits(values)
        self.addr_width = self._calculate_required_addr_width_to_access_items(values)
        padded_values = pad_with_zeros(list(values), 2**self.addr_width)
        self.hex_values = list(
            map(lambda fp: hex_representation(fp.to_hex()), padded_values)
        )
        self.resource_option = resource_option

    @staticmethod
    def _calculate_required_addr_width_to_access_items(items: Sequence) -> int:
        return max(1, math.ceil(math.log2(len(items))))

    def __call__(self) -> Code:
        template = read_text("elasticai.creator.vhdl.templates", "rom.tpl.vhd")

        code = template.format(
            rom_name=self.rom_name,
            rom_addr_bitwidth=self.addr_width,
            rom_data_bitwidth=self.data_width,
            rom_value=",".join(self.hex_values),
            rom_resource_option=f'"{self.resource_option}"',
        )

        stripped_code_lines = map(str.strip, code.splitlines())

        def not_empty(line):
            return len(line) > 0

        yield from filter(not_empty, stripped_code_lines)
