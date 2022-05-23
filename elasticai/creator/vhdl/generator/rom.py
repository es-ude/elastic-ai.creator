import math
from functools import partial

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.number_representations import hex_representation


def pad_with_zeros(numbers: list[int], target_length: int) -> list[int]:
    return numbers + [0] * (target_length - len(numbers))


class Rom:
    def __init__(
        self, rom_name: str, data_width: int, values: list[int], resource_option: str
    ):
        self.rom_name = rom_name
        self.data_width = data_width
        self.addr_width = self._calculate_required_addr_width_to_access_items(values)
        padded_values = pad_with_zeros(values, 2**self.addr_width)
        to_hex = partial(hex_representation, num_bits=data_width)
        self.hex_values = list(map(to_hex, padded_values))
        self.resource_option = resource_option

    @staticmethod
    def _calculate_required_addr_width_to_access_items(items: list) -> int:
        return max(1, math.ceil(math.log2(len(items))))

    def __call__(self):
        template = read_text(
            "elasticai.creator.vhdl.generator.templates", "rom.tpl.vhd"
        )

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
