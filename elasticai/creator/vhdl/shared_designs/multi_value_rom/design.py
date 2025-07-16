import math
from functools import partial

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.code_generation.code_abstractions import (
    multi_values_to_vhdl_binary_string,
)


class MultiValueRom:
    def __init__(
        self,
        name: str,
        data_width: int,
        values_as_integers: list[int],
        number_per_row: int,
        unroll_factor: int,
    ) -> None:
        self._name = name
        self._data_width = data_width
        self._rom_data_width = data_width * unroll_factor
        number_of_values = math.ceil(len(values_as_integers) / unroll_factor)
        self._address_width = self._bits_required_to_address_n_values(number_of_values)
        self._unroll_factor = unroll_factor
        self._number_per_row = number_per_row
        self._restructured_values = self._restructure_values(values_as_integers)
        self._values = [
            multi_values_to_vhdl_binary_string(x, self._data_width)
            for x in self._append_zeros_to_fill_addressable_memory(
                self._restructured_values
            )
        ]

    def save_to(self, destination: Path):
        template = InProjectTemplate(
            file_name="rom.tpl.vhd",
            package=module_to_package(self.__module__),
            parameters=dict(
                rom_value=self._rom_values(),
                rom_addr_bitwidth=str(self._address_width),
                rom_data_bitwidth=str(self._rom_data_width),
                name=self._name,
                resource_option="auto",
            ),
        )
        destination.as_file(".vhd").write(template)

    def _rom_values(self) -> str:
        return ",".join(self._values)

    def _restructure_values(self, values: list[int]) -> list[list[int]]:
        restructured_values = []
        num_of_rows = len(values) // (self._number_per_row * self._unroll_factor)

        for row in range(num_of_rows):
            index_start = row * self._number_per_row * self._unroll_factor
            for i in range(self._number_per_row):
                index = index_start + i
                if index < len(values):
                    grouped_values = []
                    for j in range(self._unroll_factor):
                        grouped_values.append(values[index + j * self._number_per_row])
                    grouped_values.reverse()
                    restructured_values.append(grouped_values)
        return restructured_values

    def _append_zeros_to_fill_addressable_memory(
        self, values: list[list[int]]
    ) -> list[list[int]]:
        missing_number_of_zeros = 2**self._address_width - len(values)
        return values + self._zeros(missing_number_of_zeros)

    def _bits_required_to_address_n_values(self, n: int) -> int:
        return calculate_address_width(n)

    def _zeros(self, n: int) -> list[list[int]]:
        return [[0] * self._unroll_factor for _ in range(n)]
