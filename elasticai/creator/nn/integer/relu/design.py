from pathlib import Path

# from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (  # InProjectTemplate,
    module_to_package,
)

# from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.nn.integer.design import Design
from elasticai.creator.nn.integer.template import Template, save_code
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.ports import Port


class ReLU(Design):
    def __init__(
        self, name: str, data_width: int, threshold: int, clock_option: bool = True
    ) -> None:
        super().__init__(name=name)
        self._data_width = data_width
        self._threshold = threshold
        self._clock_option = clock_option

    @property
    def port(self) -> Port:
        return create_port(x_width=self._data_width, y_width=self._data_width)

    def save_to(self, destination: Path) -> None:
        template = Template(
            package=module_to_package(self.__module__),
            file_name="relu.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                threshold=str(self._threshold),
                clock_option="true" if self._clock_option else "false",
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = Template(
            package=module_to_package(self.__module__),
            file_name="relu_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                threshold=str(self._threshold),
                clock_option="true" if self._clock_option else "false",
            ),
        )

        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)