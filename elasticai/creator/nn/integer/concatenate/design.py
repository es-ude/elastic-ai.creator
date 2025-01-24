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


class Concatenate(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        num_features: int,
        num_dimensions: int,
        m_q_1: int,
        m_q_2: int,
        m_q_1_shift: int,
        m_q_2_shift: int,
        z_x1: int,
        z_x2: int,
        z_y: int,
        work_library_name: str,
        resource_option: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._num_features = num_features
        self._num_dimensions = num_dimensions

        self._m_q_1 = m_q_1
        self._m_q_2 = m_q_2
        self._m_q_1_shift = m_q_1_shift
        self._m_q_2_shift = m_q_2_shift
        self._m_q_1_data_width = int(np.ceil(np.log2(self._m_q_1))) + 1
        self._m_q_2_data_width = int(np.ceil(np.log2(self._m_q_2))) + 1

        self._z_x1 = z_x1
        self._z_x2 = z_x2
        self._z_y = z_y

        self._work_library_name = work_library_name
        self._resource_option = resource_option

        self._m_q_data_width = max(self._m_q_1_data_width, self._m_q_2_data_width)

        self._x_count = self._num_features * self._num_dimensions
        self._y_count = self._x_count

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
            file_name="concatenate.tpl.vhd",  # TODO: write the correct VHDL template
            parameters=dict(
                name=self.name,
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                data_width=str(self._data_width),
                num_features=str(self._num_features),
                num_dimensions=str(self._num_dimensions),
                m_q_1=str(self._m_q_1),
                m_q_1_shift=str(self._m_q_1_shift),
                m_q_2=str(self._m_q_2),
                m_q_2_shift=str(self._m_q_2_shift),
                m_q_data_width=str(self._m_q_data_width),
                z_x_1=str(self._z_x1),
                z_x_2=str(self._z_x2),
                z_y=str(self._z_y),
                work_library_name=self._work_library_name,
                resource_option=self._resource_option,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        ram = Ram(name=f"{self.name}_ram")
        ram.save_to(destination)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="concatenate_tb.tpl.vhd",  # TODO: write the correct VHDL template
            parameters=dict(
                name=self.name,
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                data_width=str(self._data_width),
                num_features=str(self._num_features),
                num_dimensions=str(self._num_dimensions),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )
