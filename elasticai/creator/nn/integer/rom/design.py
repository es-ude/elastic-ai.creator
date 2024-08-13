from pathlib import Path

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port
from elasticai.creator.vhdl.shared_designs.rom.design import Rom as CreatorRom


class Rom(Design, CreatorRom):
    def __init__(
        self,
        name: str,
        data_width: int,
        values_as_integers: list[int],
        resource_option: str = "auto",
    ) -> None:
        values_as_integers = [int(x.item()) for x in values_as_integers]

        Design.__init__(self, name=name)
        CreatorRom.__init__(
            self,
            name=name,
            data_width=data_width,
            values_as_integers=values_as_integers,
        )
        self._resource_option = resource_option

    @property
    def port(self) -> Port:
        pass

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="rom.tpl.vhd",
            parameters=dict(
                rom_value=self._rom_values(),
                rom_addr_bitwidth=str(self._address_width),
                rom_data_bitwidth=str(self._data_width),
                name=self.name,
                resource_option=self._resource_option,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)
