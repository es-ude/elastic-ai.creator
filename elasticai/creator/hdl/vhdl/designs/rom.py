from functools import partial

from elasticai.creator.hdl.code_generation.abstract_base_template import (
    TemplateConfig,
    TemplateExpander,
    module_to_package,
)
from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.translatable import Path
from elasticai.creator.hdl.vhdl.code_generation.twos_complement import to_unsigned


class Rom:
    def __init__(self, name: str, data_width: int, values_as_integers: list[int]):
        self._name = name
        self._data_width = data_width
        number_of_values = len(values_as_integers)
        self._address_width = self._bits_required_to_address_n_values(number_of_values)
        self._values = self._append_zeros_to_fill_addressable_memory(
            self._values_to_unsigned_integers(values_as_integers)
        )

    def save_to(self, destination: Path):
        config = TemplateConfig(
            file_name="rom.tpl.vhd",
            package=module_to_package(self.__module__),
            parameters=dict(
                rom_value=self._rom_values(),
                rom_addr_bitwidth=str(self._address_width),
                rom_data_bitwidth=str(self._data_width),
                name=self._name,
                resource_option="auto",
            ),
        )
        expander = TemplateExpander(config)
        destination.as_file(".vhd").write_text(expander.lines())

    def _values_to_unsigned_integers(self, values: list[int]) -> list[int]:
        to_uint = partial(to_unsigned, total_bits=self._data_width)
        return list(map(to_uint, values))

    def _rom_values(self) -> str:
        values = [self._format_string_for_rom_values().format(x) for x in self._values]
        return ",".join(values)

    def _format_string_for_rom_values(self) -> str:
        return 'x"{{:0>{num_of_nibbles}x}}"'.format(
            num_of_nibbles=self._number_of_nibbles()
        )

    def _append_zeros_to_fill_addressable_memory(self, values: list[int]) -> list[int]:
        missing_number_of_zeros = 2**self._address_width - len(values)
        return values + self._zeros(missing_number_of_zeros)

    def _bits_required_to_address_n_values(self, n: int) -> int:
        return calculate_address_width(n)

    @staticmethod
    def _zeros(n: int) -> list[int]:
        return [0] * n

    def _number_of_nibbles(self) -> int:
        nibble_length = 4
        return self._data_width // nibble_length
