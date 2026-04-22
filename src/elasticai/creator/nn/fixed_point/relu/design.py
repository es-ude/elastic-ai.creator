from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design, Port


class ReLU(Design):
    def __init__(self, name: str, total_bits: int, use_clock: bool) -> None:
        super().__init__(name)
        self._total_bits = total_bits
        self._clock_option = "true" if use_clock else "false"

    @property
    def port(self) -> Port:
        return create_port(x_width=self._total_bits, y_width=self._total_bits)

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="relu.tpl.vhd",
            parameters=dict(
                layer_name=self.name,
                data_width=str(self._total_bits),
                clock_option=self._clock_option,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)
