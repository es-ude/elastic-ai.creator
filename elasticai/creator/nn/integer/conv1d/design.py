from itertools import chain

import numpy as np

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.shared_designs.rom import Rom


class Conv1d(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        weights: list[list[list[int]]],
        bias: list[int],
        m_q: int,
        m_q_shift: int,
        z_x: int,
        z_w: int,
        z_b: int,
        z_y: int,
        work_library_name: str,
        resource_option: str,
    ):
        super().__init__(name=name)

        self._data_width = data_width
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        self._m_q = m_q
        self._m_q_shift = m_q_shift
        self._m_q_data_width = int(np.ceil(np.log2(self._m_q))) + 1

        self._z_x = z_x
        self._z_w = z_w
        self._z_b = z_b
        self._z_y = z_y

        self._weights = [
            [[int(w) + self._z_w for w in kernel] for kernel in channel]
            for channel in weights
        ]
        self._bias = [int(b) + self._z_b for b in bias]

        self._work_library_name = work_library_name
        self._resource_option = resource_option

    @property
    def port(self):
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self._in_channels,
            y_count=self._out_channels,
        )

    def save_to(self, destination: Path):
        rom_name = dict(weights=f"{self.name}_w_rom", bias=f"{self.name}_b_rom")

        # VHDL Template
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="conv1d.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                in_channels=str(self._in_channels),
                out_channels=str(self._out_channels),
                kernel_size=str(self._kernel_size),
                stride=str(self._stride),
                padding=str(self._padding),
                z_x=str(self._z_x),
                z_w=str(self._z_w),
                z_b=str(self._z_b),
                z_y=str(self._z_y),
                m_q=str(self._m_q),
                m_q_shift=str(self._m_q_shift),
                m_q_data_width=str(self._m_q_data_width),
                weights_rom_name=rom_name["weights"],
                bias_rom_name=rom_name["bias"],
                work_library_name=self._work_library_name,
                resource_option=self._resource_option,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        # ROM for weights
        rom_weights = Rom(
            name=rom_name["weights"],
            data_width=self._data_width,
            values_as_integers=_flatten_params(self._weights),
        )
        rom_weights.save_to(destination.create_subpath(rom_name["weights"]))

        # ROM for biases
        rom_bias = Rom(
            name=rom_name["bias"],
            data_width=self._data_width + 1,
            values_as_integers=self._bias,
        )
        rom_bias.save_to(destination.create_subpath(rom_name["bias"]))


def _flatten_params(params: list[list[list[int]]]) -> list[int]:
    return list(chain(*[list(chain(*channel)) for channel in params]))
