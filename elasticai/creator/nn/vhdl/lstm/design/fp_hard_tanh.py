from elasticai.creator.hdl.code_generation.abstract_base_template import (
    TemplateConfig,
    TemplateExpander,
    module_to_package,
)
from elasticai.creator.hdl.design_base.design import Design, Port
from elasticai.creator.hdl.design_base.signal import Signal
from elasticai.creator.hdl.translatable import Path
from elasticai.creator.nn.two_complement_fixed_point_config import FixedPointConfig


class FPHardTanh(Design):
    def __init__(self, total_bits: int, frac_bits: int):
        super().__init__(name="hardtanh")
        self._data_width = total_bits
        fp_config = FixedPointConfig(frac_bits=frac_bits, total_bits=total_bits)
        self._template = TemplateConfig(
            package=module_to_package(self.__module__),
            file_name="fp_hard_tanh.tpl.vhd",
            parameters=dict(
                data_width=str(self._data_width),
                one=str(fp_config.as_integer(1)),
                minus_one=str(fp_config.as_integer(-1)),
            ),
        )

    def save_to(self, destination: "Path"):
        destination.as_file(".vhd").write_text(TemplateExpander(self._template).lines())

    @property
    def port(self) -> Port:
        return Port(
            incoming=[Signal("x", self._data_width)],
            outgoing=[Signal("y", self._data_width)],
        )
