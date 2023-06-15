from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.design_base.design import Design, Port
from elasticai.creator.vhdl.savable import Path


class FPReLU(Design):
    def __init__(self, name: str, data_width: int, use_clock: bool) -> None:
        super().__init__(name)
        self._data_width = data_width
        self._clock_option = "true" if use_clock else "false"

    @property
    def port(self) -> Port:
        return create_port(x_width=self._data_width, y_width=self._data_width)

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="fp_relu.tpl.vhd",
            parameters=dict(
                layer_name=self.name,
                data_width=str(self._data_width),
                clock_option=self._clock_option,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)
