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


class BatchNorm1d(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        num_dimensions: int,
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
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._num_dimensions = num_dimensions
        self._in_features = in_features
        self._out_features = out_features

        self._m_q = m_q
        self._m_q_shift = m_q_shift
        self._m_q_data_width = int(np.ceil(np.log2(self._m_q))) + 1

        self._z_x = z_x
        self._z_w = z_w
        self._z_b = z_b
        self._z_y = z_y

        self._weights = [w + self._z_w for w in weights]
        self._bias = [b + self._z_b for b in bias]

        self._work_library_name = work_library_name
        self._resource_option = resource_option

        self._x_count = self._in_features * self._num_dimensions
        self._y_count = self._out_features * self._num_dimensions

        self._x_addr_width = calculate_address_width(self._x_count)
        self._y_addr_width = calculate_address_width(self._y_count)

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_addr_width=self._x_addr_width,
            y_addr_width=self._y_addr_width,
        )

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(__name__),
            file_name="batchnorm1d.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                num_dimensions=str(self._num_dimensions),
                in_features=str(self._in_features),
                out_features=str(self._out_features),
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                m_q=str(self._m_q),
                m_q_shift=str(self._m_q_shift),
                m_q_data_width=str(self._m_q_data_width),
                z_w=str(self._z_w),
                z_b=str(self._z_b),
                z_x=str(self._z_x),
                z_y=str(self._z_y),
                work_library_name=self._work_library_name,
                resource_option=self._resource_option,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        rom_name = dict(weights=f"{self.name}_w_rom", bias=f"{self.name}_b_rom")
        rom_weights = Rom(
            name=rom_name["weights"],
            data_width=self._data_width,
            values_as_integers=self._weights,
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
            package=module_to_package(__name__),
            file_name="batchnorm1d_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                num_dimensions=str(self._num_dimensions),
                in_features=str(self._in_features),
                out_features=str(self._out_features),
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )
