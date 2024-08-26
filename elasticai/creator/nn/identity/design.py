from pathlib import Path

from elasticai.creator.file_generation.template import InProjectTemplate, save_template
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class BufferedIdentity(Design):
    def __init__(self, name: str, num_input_features: int, num_input_bits: int) -> None:
        self._num_input_features = num_input_features
        self._num_input_bits = num_input_bits
        self._address_width = calculate_address_width(num_input_features)
        super().__init__(name)

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._num_input_bits,
            y_width=self._num_input_bits,
            x_count=self._num_input_features,
            y_count=self._num_input_features,
        )

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=InProjectTemplate.module_to_package(self.__module__),
            file_name="buffered_identity.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_address_width=str(self._address_width),
                y_address_width=str(self._address_width),
                x_width=str(self._num_input_bits),
                y_width=str(self._num_input_bits),
            ),
        )
        save_template(template, destination / f"{self.name}.vhd")


class BufferlessDesign(Design):
    def __init__(self, name: str, num_input_bits: int):
        super().__init__(name)
        self._num_input_bits = num_input_bits

    @property
    def port(self) -> Port:
        return create_port(x_width=self._num_input_bits, y_width=self._num_input_bits)

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=InProjectTemplate.module_to_package(self.__module__),
            file_name="bufferless_identity.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_width=str(self._num_input_bits),
                y_width=str(self._num_input_bits),
            ),
        )
        save_template(template, destination / f"{self.name}.vhd")
