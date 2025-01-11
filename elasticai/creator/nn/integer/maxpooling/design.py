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


class MaxPooling1d(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        in_features: int,
        out_features: int,
        in_num_dimensions: int,
        out_num_dimensions: int,
        kernel_size: int,
        work_library_name: str,
        resource_option: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._in_features = in_features
        self._out_features = out_features
        self._in_num_dimensions = in_num_dimensions
        self._out_num_dimensions = out_num_dimensions
        self._kernel_size = kernel_size

        self._work_library_name = work_library_name
        self._resource_option = resource_option

        self._x_count = self._in_features * self._in_num_dimensions
        self._y_count = self._out_features * self._out_num_dimensions

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
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="maxpooling.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                data_width=str(self._data_width),
                in_features=str(self._in_features),
                out_features=str(self._out_features),
                in_num_dimensions=str(self._in_num_dimensions),
                out_num_dimensions=str(self._out_num_dimensions),
                kernel_size=str(self._kernel_size),
                work_library_name=self._work_library_name,
                resource_option=self._resource_option,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        ram = Ram(name=f"{self.name}_ram")
        ram.save_to(destination)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="maxpooling_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                data_width=str(self._data_width),
                in_features=str(self._in_features),
                out_features=str(self._out_features),
                in_num_dimensions=str(self._in_num_dimensions),
                out_num_dimensions=str(self._out_num_dimensions),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )


def _flatten_params(params: list[list[int]]) -> list[int]:
    return list(chain(*params))
