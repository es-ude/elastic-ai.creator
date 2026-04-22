from elasticai.creator.arithmetic import FxpConverter, FxpParams
from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width


class Rom:
    def __init__(
        self, name: str, data_width: int, values_as_integers: list[int]
    ) -> None:
        self._name = name
        self._data_width = data_width
        self._address_width = calculate_address_width(len(values_as_integers))
        conv = FxpConverter(
            FxpParams(total_bits=self._data_width, frac_bits=0, signed=True)
        )
        self._values = [
            conv.integer_to_binary_string_vhdl(x)
            for x in self._append_zeros_to_fill_addressable_memory(values_as_integers)
        ]

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
        destination.create_subpath(self._name).as_file(".vhd").write(template)

    def _rom_values(self) -> str:
        return ", ".join(self._values)

    def _append_zeros_to_fill_addressable_memory(self, values: list[int]) -> list[int]:
        missing_number_of_zeros = 2**self._address_width - len(values)
        return values + [0] * missing_number_of_zeros
