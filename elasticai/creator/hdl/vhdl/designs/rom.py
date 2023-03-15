from math import ceil, log2

from elasticai.creator.hdl.code_generation.abstract_base_template import (
    TemplateConfig,
    TemplateExpander,
    module_to_package,
)
from elasticai.creator.hdl.translatable import Path


class Rom:
    def __init__(
        self, name: str, data_width: int, values_as_unsigned_integers: list[int]
    ):
        self._name = name
        self._address_width = self._compute_bits_required_to_address_n_values(
            len(values_as_unsigned_integers)
        )
        self._values = self._append_zeros_to_fill_addressable_memory(
            values_as_unsigned_integers
        )
        self._data_width = data_width

    def _append_zeros_to_fill_addressable_memory(self, values: list[int]) -> list[int]:
        return values + [0] * (self._address_width**2 - len(values))

    def _compute_bits_required_to_address_n_values(self, n: int) -> int:
        return ceil(log2(n))

    def save_to(self, destination: Path):
        nibble_length = 4
        format_string = 'x"{{:0>{num_of_hex_values}x}}"'.format(
            num_of_hex_values=self._data_width // nibble_length
        )
        print(format_string)
        values = [format_string.format(x) for x in self._values]
        config = TemplateConfig(
            file_name="rom.tpl.vhd",
            package=module_to_package(self.__module__),
            parameters=dict(
                rom_value=", ".join(values),
                rom_addr_bitwidth=str(self._address_width),
                rom_data_width=str(self._data_width),
                name=self._name,
            ),
        )
        expander = TemplateExpander(config)
        destination.as_file(".vhd").write_text(expander.lines())
