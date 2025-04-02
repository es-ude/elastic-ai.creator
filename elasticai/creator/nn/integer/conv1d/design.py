from itertools import chain

import numpy as np

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.nn.integer.math_operations import (
    get_padded_count,
    get_vhdl_templates,
)
from elasticai.creator.nn.integer.ram.design import Ram
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port
from elasticai.creator.vhdl.shared_designs.rom import Rom


class Conv1d(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        kernel_size: int,
        padding: tuple[int, int] or str,
        padding_len: int,
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
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._seq_len = seq_len
        self._kernel_size = kernel_size
        self._padding = padding
        self._padding_len = padding_len

        self._m_q = m_q
        self._m_q_shift = m_q_shift
        self._m_q_data_width = (
            int(np.ceil(np.log2(self._m_q))) + 1
            if self._m_q != 0
            else 1 if self._m_q != 0 else 1
        )

        self._z_x = z_x
        self._z_w = z_w
        self._z_b = z_b
        self._z_y = z_y

        self._weights = [
            [[w + self._z_w for w in column] for column in row] for row in weights
        ]
        self._bias = [b + self._z_b for b in bias]

        self._work_library_name = work_library_name
        self._resource_option = resource_option

        self._x_count, self._y_count = get_padded_count(
            padding=self._padding,
            kernel_size=self._kernel_size,
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            seq_len=self._seq_len,
        )

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
            kernel_size=str(self._kernel_size),
            in_channels=str(self._in_channels),
            out_channels=str(self._out_channels),
            seq_len=str(self._seq_len),
            m_q_data_width=str(self._m_q_data_width),
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

        template_file_name, test_template_file_name = get_vhdl_templates(
            self._padding_len, "conv1d"
        )
        if self._padding_len != 0:
            template_parameters["padding_len"] = str(self._padding_len)

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
                out_channels=str(self._out_channels),
                seq_len=str(self._seq_len),
                kernel_size=str(self._kernel_size),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )


def _flatten_params(params):
    flat_list = []
    for p in params:
        if isinstance(p, list):
            flat_list.extend(_flatten_params(p))
        else:
            flat_list.append(p)
    return flat_list
