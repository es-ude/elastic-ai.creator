from itertools import chain

import numpy as np

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.nn.integer.ram.design import Ram
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
        m_q: int,
        m_q_shift: int,
        z_w: int,
        z_b: int,
        z_x: int,
        z_y: int,
        work_library_name: str,
        resource_option: str,
        num_dimensions: int = None,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._in_features = in_features
        self._out_features = out_features
        self._num_dimensions = num_dimensions

        self._m_q = m_q
        self._m_q_shift = m_q_shift
        self._m_q_data_width = int(np.ceil(np.log2(self._m_q))) + 1

        self._z_x = z_x
        self._z_w = z_w
        self._z_b = z_b
        self._z_y = z_y

        self._weights = [[w + self._z_w for w in row] for row in weights]
        self._bias = [b + self._z_b for b in bias]

        self._work_library_name = work_library_name
        self._resource_option = resource_option

        if self._num_dimensions is None:
            self._x_count = self._in_features
            self._y_count = self._out_features
        else:
            self._x_count = self._in_features * self._num_dimensions
            self._y_count = self._out_features * self._num_dimensions

        self._x_addr_width = calculate_address_width(self._x_count)
        self._y_addr_width = calculate_address_width(self._y_count)

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self._x_count,
            y_count=self._y_count,
        )

    def save_to(self, destination: Path) -> None:
        rom_name = dict(weights=f"{self.name}_w_rom", bias=f"{self.name}_b_rom")

        template_parameters = dict(
            name=self.name,
            x_addr_width=str(self._x_addr_width),
            y_addr_width=str(self._y_addr_width),
            data_width=str(self._data_width),
            m_q_data_width=str(self._m_q_data_width),
            in_features=str(self._in_features),
            out_features=str(self._out_features),
            z_x=str(self._z_x),
            z_w=str(self._z_w),
            z_b=str(self._z_b),
            z_y=str(self._z_y),
            m_q=str(self._m_q),
            m_q_shift=str(self._m_q_shift),
            weights_rom_name=rom_name["weights"],
            bias_rom_name=rom_name["bias"],
            work_library_name=self._work_library_name,
            resource_option=self._resource_option,
        )

        test_template_parameters = dict(
            name=self.name,
            x_addr_width=str(self._x_addr_width),
            y_addr_width=str(self._y_addr_width),
            data_width=str(self._data_width),
            in_features=str(self._in_features),
            out_features=str(self._out_features),
            work_library_name=self._work_library_name,
        )

        if self._num_dimensions is None:
            file_name = "linear.tpl.vhd"
            test_file_name = "linear_tb.tpl.vhd"
        else:
            if self._in_features != 1:
                file_name = "linear_2d.tpl.vhd"
            else:
                file_name = "linear_2d_feature1.tpl.vhd"
            test_file_name = "linear_2d_tb.tpl.vhd"
            template_parameters["num_dimensions"] = str(self._num_dimensions)
            test_template_parameters["num_dimensions"] = str(self._num_dimensions)

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=file_name,
            parameters=template_parameters,
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        rom_weights = Rom(
            name=rom_name["weights"],
            data_width=self._data_width,
            values_as_integers=_flatten_params(self._weights),
        )
        rom_weights.save_to(destination.create_subpath(rom_name["weights"]))

        rom_bias = Rom(
            name=rom_name["bias"],
            data_width=(self._data_width + 1) * 2,
            values_as_integers=self._bias,
        )
        rom_bias.save_to(destination.create_subpath(rom_name["bias"]))

        ram = Ram(name=f"{self.name}_ram")
        ram.save_to(destination)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=test_file_name,
            parameters=test_template_parameters,
        )
        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )


def _flatten_params(params: list[list[int]]) -> list[int]:
    return list(chain(*params))
