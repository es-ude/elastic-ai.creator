from elasticai.creator.hdl.auto_wire_protocols.buffered import (
    create_port_for_buffered_design,
)
from elasticai.creator.hdl.auto_wire_protocols.bufferless import (
    create_port_for_bufferless_design,
)
from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.code_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.design_base.ports import Port
from elasticai.creator.hdl.savable import Path


class BufferedIdentity(Design):
    def __init__(self, name: str, num_input_features: int, num_input_bits: int) -> None:
        self._num_input_features = num_input_features
        self._num_input_bits = num_input_bits
        self._address_width = calculate_address_width(num_input_features)
        super().__init__(name)

    @property
    def port(self) -> Port:
        return create_port_for_bufferless_design(
            x_width=self._num_input_bits,
            y_width=self._num_input_bits,
            x_count=self._num_input_features,
            y_count=self._num_input_features,
        )

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="buffered_identity.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_address_width=str(self._address_width),
                y_address_width=str(self._address_width),
                x_width=str(self._num_input_bits),
                y_width=str(self._num_input_bits),
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)


class BaseIdentity(Design):
    def __init__(self, name: str, num_input_bits: int):
        super().__init__(name)
        self._num_input_bits = num_input_bits

    @property
    def port(self) -> Port:
        return create_port_for_bufferless_design(
            x_width=self._num_input_bits, y_width=self._num_input_bits
        )

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="buffered_identity.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_width=str(self._num_input_bits),
                y_width=str(self._num_input_bits),
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)
