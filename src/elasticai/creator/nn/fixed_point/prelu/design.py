from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design, Port
from elasticai.creator_plugins.act_func.utils import load_and_plugin


class PReLU(Design):
    def __init__(
        self, name: str, total_bits: int, frac_bits: int, weights: list[int]
    ) -> None:
        super().__init__(name)
        self._total_bits = total_bits
        self._frac_bits = frac_bits
        self._weights = weights

    @property
    def port(self) -> Port:
        return create_port(x_width=self._total_bits, y_width=self._total_bits)

    def save_to(self, destination: Path, take_vhdl: bool = True) -> None:
        if len(self._weights) > 1:
            raise NotImplementedError("Actual layer only supports one weight")

        def _save_to_vhdl(destination: Path) -> None:
            template = InProjectTemplate(
                package=module_to_package(self.__module__),
                file_name="prelu.tpl.vhd",
                parameters=dict(
                    layer_name=self.name,
                    data_width=str(self._total_bits),
                    frac_width=str(self._frac_bits),
                    scaling=str(self._weights[0]),
                ),
            )
            destination.create_subpath(self.name).as_file(".vhd").write(template)

        def _save_to_verilog(destination: Path):
            load_and_plugin(
                type="prelu",
                id=self.name,
                params={
                    "BITWIDTH": self._total_bits,
                    "FRACWIDTH": self._frac_bits,
                    "SCALING": self._weights[0],
                },
                packages=["act_func"],
                path2save=str(destination),
            )

        if take_vhdl:
            _save_to_vhdl(destination)
        else:
            _save_to_verilog(destination)
