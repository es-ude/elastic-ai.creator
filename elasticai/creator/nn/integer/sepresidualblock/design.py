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


class SeparableResidualBlock(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        seq_len: int,
        depthwise_conv1d_1: object,
        pointwise_conv1dbn_1: object,
        pointwise_conv1dbn_1_relu: object,
        depthwise_conv1d_2: object,
        pointwise_conv1dbn_2: object,
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

        self._depthwise_conv1d_1 = depthwise_conv1d_1
        self._pointwise_conv1dbn_1 = pointwise_conv1dbn_1
        self._pointwise_conv1dbn_1_relu = pointwise_conv1dbn_1_relu
        self._depthwise_conv1d_2 = depthwise_conv1d_2
        self._pointwise_conv1dbn_2 = pointwise_conv1dbn_2
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
        depthwise_conv1d_1_deisgn = self._depthwise_conv1d_1.create_design(
            name=self._depthwise_conv1d_1.name
        )
        depthwise_conv1d_1_deisgn.save_to(destination)

        pointwise_conv1dbn_1_deisgn = self._pointwise_conv1dbn_1.create_design(
            name=self._pointwise_conv1dbn_1.name
        )
        pointwise_conv1dbn_1_deisgn.save_to(destination)

        pointwise_conv1dbn_1_relu_deisgn = (
            self._pointwise_conv1dbn_1_relu.create_design(
                name=self._pointwise_conv1dbn_1_relu.name
            )
        )
        pointwise_conv1dbn_1_relu_deisgn.save_to(destination)

        depthwise_conv1d_2_deisgn = self._depthwise_conv1d_2.create_design(
            name=self._depthwise_conv1d_2.name
        )
        depthwise_conv1d_2_deisgn.save_to(destination)

        pointwise_conv1dbn_2_deisgn = self._pointwise_conv1dbn_2.create_design(
            name=self._pointwise_conv1dbn_2.name
        )
        pointwise_conv1dbn_2_deisgn.save_to(destination)

        shortcut_deisgn = self._shortcut.create_design(name=self._shortcut.name)
        shortcut_deisgn.save_to(destination)

        add_deisgn = self._add.create_design(name=self._add.name)
        add_deisgn.save_to(destination)

        relu_deisgn = self._relu.create_design(name=self._relu.name)
        relu_deisgn.save_to(destination)

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="sepresidualblock.tpl.vhd",
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
            file_name="sepresidualblock_tb.tpl.vhd",
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
