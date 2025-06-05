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


class MatrixMulti(Design):
    def __init__(
        self,
        name: str,
        is_score_mode: str,
        data_width: int,
        x_1_dim_a: int,
        x_1_dim_b: int,
        x_1_dim_c: int,
        x_2_dim_a: int,
        x_2_dim_b: int,
        x_2_dim_c: int,
        y_dim_a: int,
        y_dim_b: int,
        y_dim_c: int,
        m_q: int,
        m_q_shift: int,
        z_x_1: int,
        z_x_2: int,
        z_y: int,
        work_library_name: str,
        resource_option: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._is_score_mode = is_score_mode
        self._x_1_dim_a = x_1_dim_a
        self._x_1_dim_b = x_1_dim_b
        self._x_1_dim_c = x_1_dim_c
        self._x_2_dim_a = x_2_dim_a
        self._x_2_dim_b = x_2_dim_b
        self._x_2_dim_c = x_2_dim_c
        self._y_dim_a = y_dim_a
        self._y_dim_b = y_dim_b
        self._y_dim_c = y_dim_c

        self._m_q = m_q
        self._m_q_shift = m_q_shift
        self._m_q_data_width = int(np.ceil(np.log2(self._m_q))) + 1

        self._z_x_1 = z_x_1
        self._z_x_2 = z_x_2
        self._z_y = z_y

        self._work_library_name = work_library_name
        self._resource_option = resource_option

        self._x_1_count = self._x_1_dim_a * self._x_1_dim_b * self._x_1_dim_c
        self._x_2_count = self._x_2_dim_a * self._x_2_dim_b * self._x_2_dim_c
        self._y_count = self._y_dim_a * self._y_dim_b * self._y_dim_c

        self._x_1_addr_width = calculate_address_width(self._x_1_count)
        self._x_2_addr_width = calculate_address_width(self._x_2_count)
        self._y_addr_width = calculate_address_width(self._y_count)

    @property
    def port(self) -> Port:
        return create_port(
            x_1_width=self._data_width,
            x_2_width=self._data_width,
            y_width=self._data_width,
            x_1_count=self._x_1_count,
            x_2_count=self._x_2_count,
            y_count=self._y_count,
        )

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="matrixmulti.tpl.vhd",
            parameters=dict(
                name=self.name,
                is_score_mode=self._is_score_mode,
                x_1_addr_width=str(self._x_1_addr_width),
                x_2_addr_width=str(self._x_2_addr_width),
                y_addr_width=str(self._y_addr_width),
                data_width=str(self._data_width),
                x_1_dim_a=str(self._x_1_dim_a),
                x_1_dim_b=str(self._x_1_dim_b),
                x_1_dim_c=str(self._x_1_dim_c),
                x_2_dim_a=str(self._x_2_dim_a),
                x_2_dim_b=str(self._x_2_dim_b),
                x_2_dim_c=str(self._x_2_dim_c),
                y_dim_a=str(self._y_dim_a),
                y_dim_b=str(self._y_dim_b),
                y_dim_c=str(self._y_dim_c),
                m_q=str(self._m_q),
                m_q_shift=str(self._m_q_shift),
                m_q_data_width=str(self._m_q_data_width),
                z_x_1=str(self._z_x_1),
                z_x_2=str(self._z_x_2),
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
            file_name="matrixmulti_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_1_addr_width=str(self._x_1_addr_width),
                x_2_addr_width=str(self._x_2_addr_width),
                y_addr_width=str(self._y_addr_width),
                data_width=str(self._data_width),
                x_1_dim_a=str(self._x_1_dim_a),
                x_1_dim_b=str(self._x_1_dim_b),
                x_1_dim_c=str(self._x_1_dim_c),
                x_2_dim_a=str(self._x_2_dim_a),
                x_2_dim_b=str(self._x_2_dim_b),
                x_2_dim_c=str(self._x_2_dim_c),
                y_dim_a=str(self._y_dim_a),
                y_dim_b=str(self._y_dim_b),
                y_dim_c=str(self._y_dim_c),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )
