from pathlib import Path

from elasticai.creator.file_generation.v2.template import (
    InProjectTemplate,
    save_template,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design, Port


class HardSigmoid(Design):
    def __init__(
        self,
        name: str,
        total_bits: int,
        frac_bits: int,
        one: int,
        zero_threshold: int,
        one_threshold: int,
        slope: int,
        y_intercept: int,
    ) -> None:
        super().__init__(name)
        self._total_bits = total_bits
        self._frac_bits = frac_bits
        self._one = one
        self._zero_threshold = zero_threshold
        self._one_threshold = one_threshold
        self._slope = slope
        self._y_intercept = y_intercept

    @property
    def port(self) -> Port:
        return create_port(x_width=self._total_bits, y_width=self._total_bits)

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=InProjectTemplate.module_to_package(self.__module__),
            file_name="hard_sigmoid.tpl.vhd",
            parameters=dict(
                layer_name=self.name,
                data_width=str(self._total_bits),
                frac_width=str(self._frac_bits),
                one=str(self._one),
                zero_threshold=str(self._zero_threshold),
                one_threshold=str(self._one_threshold),
                slope=str(self._slope),
                y_intercept=str(self._y_intercept),
            ),
        )
        save_template(template, destination / f"{self.name}.vhd")
