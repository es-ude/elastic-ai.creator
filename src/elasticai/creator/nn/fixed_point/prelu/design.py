from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design, Port


class PReLU(Design):
    def __init__(
        self, name: str, total_bits: int, frac_bits: int, weights: list
    ) -> None:
        super().__init__(name)
        self._total_bits = total_bits
        self._frac_bits = frac_bits
        self._weights = weights

    @property
    def port(self) -> Port:
        return create_port(x_width=self._total_bits, y_width=self._total_bits)

    def save_to(self, destination: Path) -> None:
        if len(self._weights) > 1:
            raise NotImplementedError("Actual layer only supports one weight")

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="prelu.tpl.vhd",
            parameters=dict(
                layer_name=self.name,
                data_width=str(self._total_bits),
                scaling=self._weights,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)
