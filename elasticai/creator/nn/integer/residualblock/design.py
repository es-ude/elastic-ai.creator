from itertools import chain

import numpy as np

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class ResidualBlock(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        seq_len: int,
        conv1dbn_0: object,
        conv1dbn_0_relu: object,
        conv1dbn_1: object,
        shortcut: object,
        add: object,
        relu: object,
        work_library_name: str,
    ):
        super().__init__(name=name)

        self._data_width = data_width
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._seq_len = seq_len
        self._kernel_size = kernel_size
        self._work_library_name = work_library_name

        self._conv1dbn_0 = conv1dbn_0
        self._conv1dbn_0_relu = conv1dbn_0_relu
        self._conv1dbn_1 = conv1dbn_1
        self._shortcut = shortcut
        self._add = add
        self._relu = relu

        self._x_count = self._in_channels * self._seq_len
        self._y_count = self._out_channels * (self._seq_len - self._kernel_size + 1)

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
        conv1dbn_0_deisgn = self._conv1dbn_0.create_design(name=self._conv1dbn_0.name)
        conv1dbn_0_deisgn.save_to(destination.create_subpath(self._conv1dbn_0.name))

        conv1dbn_0_relu_deisgn = self._conv1dbn_0_relu.create_design(
            name=self._conv1dbn_0_relu.name
        )
        conv1dbn_0_relu_deisgn.save_to(
            destination.create_subpath(self._conv1dbn_0_relu.name)
        )

        conv1dbn_1_deisgn = self._conv1dbn_1.create_design(name=self._conv1dbn_1.name)
        conv1dbn_1_deisgn.save_to(destination.create_subpath(self._conv1dbn_1.name))

        if len(self._shortcut) > 0:
            for submodule in self._shortcut:
                submodule_design = submodule.create_design(name=submodule.name)
                submodule_design.save_to(destination.create_subpath(submodule.name))

        add_design = self._add.create_design(name=self._add.name)
        add_design.save_to(destination.create_subpath(self._add.name))

        relu_design = self._relu.create_design(name=self._relu.name)
        relu_design.save_to(destination.create_subpath(self._relu.name))

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="residualblock.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="residualblock_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                in_channels=str(self._in_channels),
                out_channels=str(self._out_channels),
                seq_len=str(self._seq_len),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )
