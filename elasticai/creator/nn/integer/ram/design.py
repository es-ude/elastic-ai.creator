from itertools import chain

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.nn.integer.design import Design
from elasticai.creator.nn.integer.rom.design import Rom

# from elasticai.creator.nn.integer.template import Template, save_code
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class Ram(Design):
    def __init__(self, name: str):
        # super().__init__(name=name)
        self.name = name

    @property
    def port(self) -> Port:
        pass

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="dual_port_2_clock_ram.tpl.vhd",
            parameters=dict(name=self.name),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)
