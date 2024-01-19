from functools import partial

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.code_generation.code_abstractions import (
    to_vhdl_binary_string,
)


class Rom:
    def __init__(
        self, name: str, data_width: int, values_as_integers: list[int]
    ) -> None:
        self._name = name
        self._data_width = data_width
        number_of_values = len(values_as_integers)
        self._address_width = self._bits_required_to_address_n_values(number_of_values)
        self._values = [to_vhdl_binary_string(x, self._data_width) for x in
                        self._append_zeros_to_fill_addressable_memory(values_as_integers)]

    def save_to(self, destination: Path):
        template = InProjectTemplate(
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
        destination.as_file(".vhd").write(template)

    def _rom_values(self) -> str:
        return ",".join(self._values)

    def _append_zeros_to_fill_addressable_memory(self, values: list[int]) -> list[int]:
        missing_number_of_zeros = 2**self._address_width - len(values)
        return values + _zeros(missing_number_of_zeros)

    def _bits_required_to_address_n_values(self, n: int) -> int:
        return calculate_address_width(n)


def _zeros(n: int) -> list[int]:
    return [0] * n


