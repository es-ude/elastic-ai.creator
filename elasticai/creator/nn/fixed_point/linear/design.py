from itertools import chain

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port
from elasticai.creator.vhdl.shared_designs.rom import Rom


class LinearDesign(Design):
    def __init__(
        self,
        *,
        in_feature_num: int,
        out_feature_num: int,
        total_bits: int,
        frac_bits: int,
        weights: list[list[int]],
        bias: list[int],
        name: str,
        work_library_name: str = "work",
        resource_option: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self._name = name
        self.weights = weights
        self.bias = bias
        self._in_feature_num = in_feature_num
        self._out_feature_num = out_feature_num
        self.work_library_name = work_library_name
        self.resource_option = resource_option
        self._frac_width = frac_bits
        self._data_width = total_bits
        self.x_addr_width = self.port["x_address"].width
        self.y_addr_width = self.port["y_address"].width

    @property
    def name(self):
        return self._name

    @property
    def in_feature_num(self) -> int:
        return self._in_feature_num

    @property
    def out_feature_num(self) -> int:
        return self._out_feature_num

    @property
    def frac_width(self) -> int:
        return self._frac_width

    @property
    def data_width(self) -> int:
        return self._data_width

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

    def save_to(self, destination: Path):
        rom_name = f"{self.name}_rom"
        rom_params = list(
            chain.from_iterable([a + [b] for a, b in zip(self.weights, self.bias)])
        )

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="linear.tpl.vhd",
            parameters=dict(
                layer_name=self.name,
                params_rom_name=rom_name,
                work_library_name=self.work_library_name,
                resource_option=f'"{self.resource_option}"',
                log2_max_value="31",
                **self._template_parameters(),
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        Rom(
            name=rom_name,
            data_width=self.data_width,
            values_as_integers=rom_params,
        ).save_to(destination)
