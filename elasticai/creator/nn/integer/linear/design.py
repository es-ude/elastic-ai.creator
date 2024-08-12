from itertools import chain

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.nn.integer.design import Design
from elasticai.creator.nn.integer.rom.design import Rom

# from elasticai.creator.nn.integer.template import Template, save_code
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port

# from elasticai.creator.vhdl.shared_designs.rom import Rom


class Linear(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        in_features: int,
        out_features: int,
        weights: list[list[int]],
        bias: list[int],
        scaler: int,
        shift: int,
        z_w: int,
        z_b: int,
        z_x: int,
        z_y: int,
        work_library_name: str = "work",
        resource_option: str = "auto",
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._in_features = in_features
        self._out_features = out_features

        self._z_x = z_x.item()
        self._z_w = z_w.item()
        self._z_b = z_b.item()
        self._z_y = z_y.item()

        self._scaler = scaler.item()
        self._shift = shift.item()

        self._weights = weights
        self._bias = bias

        self._x_addr_width = calculate_address_width(self._in_features)
        self._y_addr_width = calculate_address_width(self._out_features)

        self._work_library_name = work_library_name
        self._resource_option = resource_option

        # assert self._scaler < 2**15, "scaler should be less than 2^15"

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self._in_features,
            y_count=self._out_features,
        )

    def save_to(self, destination: Path) -> None:
        rom_name = dict(weights=f"{self.name}_w_rom", bias=f"{self.name}_b_rom")

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="linear.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                data_width=str(self._data_width),
                in_features=str(self._in_features),
                out_features=str(self._out_features),
                z_x=str(self._z_x),
                z_w=str(self._z_w),
                z_b=str(self._z_b),
                z_y=str(self._z_y),
                scaler=str(self._scaler),
                shift=str(self._shift),
                weights_rom_name=rom_name["weights"],
                bias_rom_name=rom_name["bias"],
                resource_option=self._resource_option,
            ),
        )
        # save the template to the destination
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        rom_weights = Rom(
            name=rom_name["weights"],
            data_width=self._data_width,
            values_as_integers=_flatten_params(self._weights),
        )
        rom_weights.save_to(destination)

        rom_bias = Rom(
            name=rom_name["bias"],
            data_width=(self._data_width + 1)
            * 2,  # Note: bias is 2x wider, check the drawio
            values_as_integers=self._bias,
        )
        rom_bias.save_to(destination)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="linear_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                in_features=str(self._in_features),
                out_features=str(self._out_features),
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)


def _flatten_params(params: list[list[int]]) -> list[int]:
    return list(chain(*params))
