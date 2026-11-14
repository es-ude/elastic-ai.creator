from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design, Port
from elasticai.creator_plugins.act_func.utils import load_and_plugin


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

    def save_to(self, destination: Path, take_vhdl: bool = True) -> None:
        def _save_to_vhdl(destination: Path) -> None:
            template = InProjectTemplate(
                package=module_to_package(self.__module__),
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
            destination.create_subpath(self.name).as_file(".vhd").write(template)

        def _save_to_verilog(destination: Path) -> None:
            load_and_plugin(
                type="hardsigmoid",
                id=self.name,
                params={
                    "BITWIDTH_IN": self._total_bits,
                    "BITWIDTH_OUT": self._total_bits,
                    "BIAS": self._y_intercept,
                    "XMAX_VAL": self._one_threshold,
                    "YMAX_VAL": self._one,
                    "XMIN_VAL": self._zero_threshold,
                    "YMIN_VAL": 0,
                },
                packages=["act_func"],
                path2save=str(destination),
            )

        if take_vhdl:
            _save_to_vhdl(destination)
        else:
            _save_to_verilog(destination)
