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
from elasticai.creator.vhdl.shared_designs.rom import Rom


class DepthConv1d(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        in_channels: int,
        seq_len: int,
        padding: int,
        kernel_size: int,
        weights: list[list[int]],
        bias: list[int],
        m_q: int,
        m_q_shift: int,
        z_x: int,
        z_w: int,
        z_b: int,
        z_y: int,
        work_library_name: str,
        resource_option: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._in_channels = in_channels
        self._seq_len = seq_len
        self._kernel_size = kernel_size
        self._padding = padding

        self._weights = [[w + z_w for w in row] for row in weights]
        self._bias = [b + z_b for b in bias]

        self._m_q = m_q
        self._m_q_shift = m_q_shift
        self._m_q_data_width = int(np.ceil(np.log2(self._m_q))) + 1

        self._z_x = z_x
        self._z_w = z_w
        self._z_b = z_b
        self._z_y = z_y

        self._work_library_name = work_library_name
        self._resource_option = resource_option

        self._x_count = self._in_channels * self._seq_len
        if self._padding == 0:
            self._y_count = self._in_channels * (self._seq_len - self._kernel_size + 1)
        elif self._padding == 1:  # padding zero and padding to same
            self._y_count = self._x_count

        self._x_addr_width = calculate_address_width(self._x_count)
        self._y_addr_width = calculate_address_width(self._y_count)

    @property
    def port(self):
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
            in_channels=str(self._in_channels),
            sef_len=str(self._seq_len),
            kernel_size=str(self._kernel_size),
            m_q=str(self._m_q),
            m_q_shift=str(self._m_q_shift),
            z_x=str(self._z_x),
            z_w=str(self._z_w),
            z_b=str(self._z_b),
            z_y=str(self._z_y),
            m_q_data_width=str(self._m_q_data_width),
            weights_rom_name=rom_name["weights"],
            bias_rom_name=rom_name["bias"],
            work_library_name=self._work_library_name,
            resource_option=self._resource_option,
        )

        if self._padding == 0:
            template_file_name = "depthconv1d_not_padding.tpl.vhd"
            test_template_file_name = "depthconv1d_not_padding_tb.tpl.vhd"
        elif self._padding == 1:
            template_file_name = "depthconv1d_zero_padding.tpl.vhd"
            test_template_file_name = "depthconv1d_zero_padding_tb.tpl.vhd"
            template_parameters["padding"] = int(self._padding)
        else:
            raise ValueError("padding must be 0 or 1")

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=template_file_name,
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
            file_name=test_template_file_name,
            parameters=dict(
                name=self.name,
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                data_width=str(self._data_width),
                in_channels=str(self._in_channels),
                sef_len=str(self._seq_len),
                kernel_size=str(self._kernel_size),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )


def _flatten_params(params: list[list[int]]) -> list[int]:
    return list(chain(*params))
