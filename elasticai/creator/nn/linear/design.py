from itertools import chain
from typing import Optional

from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.design_base.design import Design
from elasticai.creator.vhdl.design_base.ports import Port
from elasticai.creator.vhdl.designs.rom import Rom
from elasticai.creator.vhdl.savable import Path


class FPLinear(Design):
    def __init__(
        self,
        *,
        in_feature_num: int,
        out_feature_num: int,
        total_bits: int,
        frac_bits: int,
        weights: list[list[int]],
        bias: list[int],
        work_library_name: str = "work",
        resource_option: str = "auto",
        name: Optional[str] = None,
    ):
        super().__init__(
            name="fp_linear" if name is None else name,
        )
        self.weights = weights
        self.bias = bias
        self.in_feature_num = in_feature_num
        self.out_feature_num = out_feature_num
        self.work_library_name = work_library_name
        self.resource_option = resource_option
        self.frac_width = frac_bits
        self.data_width = total_bits
        self.x_addr_width = self.port["x_address"].width
        self.y_addr_width = self.port["y_address"].width

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self.data_width,
            y_width=self.data_width,
            x_count=self.in_feature_num,
            y_count=self.out_feature_num,
        )

    def _template_parameters(self) -> dict[str, str]:
        return dict(
            (key, str(getattr(self, key)))
            for key in (
                "data_width",
                "frac_width",
                "x_addr_width",
                "y_addr_width",
                "in_feature_num",
                "out_feature_num",
            )
        )

    @staticmethod
    def _flatten_params(params: list[list[int]]) -> list[int]:
        return list(chain(*params))

    def save_to(self, destination: Path):
        rom_name = dict(weights=f"{self.name}_w_rom", bias=f"{self.name}_b_rom")

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="fp_linear.tpl.vhd",
            parameters=dict(
                layer_name=self.name,
                weights_rom_name=rom_name["weights"],
                bias_rom_name=rom_name["bias"],
                work_library_name=self.work_library_name,
                resource_option=f'"{self.resource_option}"',
                **self._template_parameters(),
            ),
        )
        destination.create_subpath(self.name).as_file(f".vhd").write(template)

        weights_rom = Rom(
            name=rom_name["weights"],
            data_width=self.data_width,
            values_as_integers=self._flatten_params(self.weights),
        )
        weights_rom.save_to(destination.create_subpath(rom_name["weights"]))

        bias_rom = Rom(
            name=rom_name["bias"],
            data_width=self.data_width,
            values_as_integers=self.bias,
        )
        bias_rom.save_to(destination.create_subpath(rom_name["bias"]))
