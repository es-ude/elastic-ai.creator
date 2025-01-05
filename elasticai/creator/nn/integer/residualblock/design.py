from itertools import chain

import numpy as np

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design


class ResidualBlock(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        weights1: list[list[int]],
        weights2: list[list[int]],
        shortcut_weights: list[list[int]],
        biases1: list[int],
        biases2: list[int],
        shortcut_biases: list[int],
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
        self._out_channels = out_channels
        self._kernel_size = kernel_size

        self._m_q = m_q
        self._m_q_shift = m_q_shift
        self._m_q_data_width = int(np.ceil(np.log2(self._m_q))) + 1

        self._z_x = z_x
        self._z_w = z_w
        self._z_b = z_b
        self._z_y = z_y

        self._weights1 = weights1
        self._weights2 = weights2
        self._shortcut_weights = shortcut_weights
        self._biases1 = biases1
        self._biases2 = biases2
        self._shortcut_biases = shortcut_biases

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

    def save_to(self, destination: Path) -> None:
        # Save ROM for weights and biases
        rom_weights1 = self._save_rom(
            destination, f"{self.name}_weights1", self._weights1
        )
        rom_weights2 = self._save_rom(
            destination, f"{self.name}_weights2", self._weights2
        )
        rom_shortcut = self._save_rom(
            destination, f"{self.name}_shortcut_weights", self._shortcut_weights
        )
        rom_biases1 = self._save_rom(destination, f"{self.name}_biases1", self._biases1)
        rom_biases2 = self._save_rom(destination, f"{self.name}_biases2", self._biases2)
        rom_shortcut_biases = self._save_rom(
            destination, f"{self.name}_shortcut_biases", self._shortcut_biases
        )

        # Save the design template
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="residual_block.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=self._data_width,
                in_channels=self._in_channels,
                out_channels=self._out_channels,
                kernel_size=self._kernel_size,
                z_x=self._z_x,
                z_w=self._z_w,
                z_b=self._z_b,
                z_y=self._z_y,
                weights1_rom=rom_weights1,
                weights2_rom=rom_weights2,
                shortcut_weights_rom=rom_shortcut,
                biases1_rom=rom_biases1,
                biases2_rom=rom_biases2,
                shortcut_biases_rom=rom_shortcut_biases,
                work_library_name=self._work_library_name,
                resource_option=self._resource_option,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

    def _save_rom(self, destination: Path, rom_name: str, values: list) -> str:
        rom_path = destination.create_subpath(rom_name)
        rom = Rom(
            name=rom_name,
            data_width=self._data_width,
            values_as_integers=chain(*values),
        )
        rom.save_to(rom_path)
        return rom_name
