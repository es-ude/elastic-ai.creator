from itertools import chain

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port
from elasticai.creator.vhdl.shared_designs.rom import Rom


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
        zero_point_weights: int,
        zero_point_bias: int,
        zero_point_inputs: int,
        zero_point_outputs: int,
        work_library_name: str = "work",
        resource_option: str = "auto",
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._in_features = in_features
        self._out_features = out_features
        self._weights = weights
        self._bias = bias
        self._scaler = scaler
        self._shift = shift
        self._zero_point_weights = zero_point_weights
        self._zero_point_bias = zero_point_bias
        self._zero_point_inputs = zero_point_inputs
        self._resource_option = resource_option
        self._zero_point_outputs = zero_point_outputs

        self._work_library_name = work_library_name

        self._x_addr_width = calculate_address_width(self._in_features)
        self._y_addr_width = calculate_address_width(self._out_features)

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
                work_library_name=self._work_library_name,
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                data_width=str(self._data_width),
                in_feature_num=str(self._in_features),
                out_feature_num=str(self._out_features),
                resource_option=self._resource_option,
                scaler=str(self._scaler),
                shift=str(self._shift),
                z_w=str(self._zero_point_weights),
                z_b=str(self._zero_point_bias),
                z_x=str(self._zero_point_inputs),
                z_y=str(self._zero_point_outputs),
                weights_rom_name=rom_name["weights"],
                bias_rom_name=rom_name["bias"],
            ),
        )
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


def _flatten_params(params: list[list[int]]) -> list[int]:
    return list(chain(*params))
