from itertools import chain

import numpy as np

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class SepResidualBlock(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        depthconv1d_0: object,
        pointconv1dbn_0: object,
        pointconv1dbn_0_relu: object,
        depthconv1d_1: object,
        pointconv1dbn_1: object,
        shortcut: object,
        add: object,
        relu: object,
        work_library_name: str,
    ):
        super().__init__(name=name)

        self._data_width = data_width
        self._work_library_name = work_library_name

        self._depthconv1d_0 = depthconv1d_0
        self._pointconv1dbn_0 = pointconv1dbn_0
        self._pointconv1dbn_0_relu = pointconv1dbn_0_relu
        self._depthconv1d_1 = depthconv1d_1
        self._pointconv1dbn_1 = pointconv1dbn_1
        self._shortcut = shortcut
        self._add = add
        self._relu = relu

        self.depthconv1d_0_deisgn = self._depthconv1d_0.create_design(
            name=self._depthconv1d_0.name
        )
        self.pointconv1dbn_0_deisgn = self._pointconv1dbn_0.create_design(
            name=self._pointconv1dbn_0.name
        )
        self.pointconv1dbn_0_relu_deisgn = self._pointconv1dbn_0_relu.create_design(
            name=self._pointconv1dbn_0_relu.name
        )
        self.depthconv1d_1_deisgn = self._depthconv1d_1.create_design(
            name=self._depthconv1d_1.name
        )
        self.pointconv1dbn_1_deisgn = self._pointconv1dbn_1.create_design(
            name=self._pointconv1dbn_1.name
        )
        self.shortcut_deisgn = self._shortcut.create_design(name=self._shortcut.name)
        self.add_deisgn = self._add.create_design(name=self._add.name)
        self.relu_deisgn = self._relu.create_design(name=self._relu.name)

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self.depthconv1d_0_deisgn._x_count,
            y_count=self.add_deisgn._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self.depthconv1d_0_deisgn.save_to(
            destination.create_subpath(self._depthconv1d_0.name)
        )
        self.pointconv1dbn_0_deisgn.save_to(
            destination.create_subpath(self._pointconv1dbn_0.name)
        )
        self.pointconv1dbn_0_relu_deisgn.save_to(
            destination.create_subpath(self._pointconv1dbn_0_relu.name)
        )
        self.depthconv1d_1_deisgn.save_to(
            destination.create_subpath(self._depthconv1d_1.name)
        )
        self.pointconv1dbn_1_deisgn.save_to(
            destination.create_subpath(self._pointconv1dbn_1.name)
        )
        self.shortcut_deisgn.save_to(destination.create_subpath(self._shortcut.name))
        self.add_deisgn.save_to(destination.create_subpath(self._add.name))
        self.relu_deisgn.save_to(destination.create_subpath(self._relu.name))

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="sepresidualblock.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                depthconv1d_0_x_addr_width=str(self.depthconv1d_0_deisgn._x_addr_width),
                depthconv1d_0_y_addr_width=str(self.depthconv1d_0_deisgn._y_addr_width),
                pointconv1dbn_0_x_addr_width=str(
                    self.pointconv1dbn_0_deisgn._x_addr_width
                ),
                pointconv1dbn_0_y_addr_width=str(
                    self.pointconv1dbn_0_deisgn._y_addr_width
                ),
                depthconv1d_1_x_addr_width=str(self.depthconv1d_1_deisgn._x_addr_width),
                depthconv1d_1_y_addr_width=str(self.depthconv1d_1_deisgn._y_addr_width),
                pointconv1dbn_1_x_addr_width=str(
                    self.pointconv1dbn_1_deisgn._x_addr_width
                ),
                pointconv1dbn_1_y_addr_width=str(
                    self.pointconv1dbn_1_deisgn._y_addr_width
                ),
                shortcut_conv1d_x_addr_width=str(self.shortcut_deisgn._x_addr_width),
                shortcut_conv1d_y_addr_width=str(self.shortcut_deisgn._y_addr_width),
                add_x_addr_width=str(self.add_deisgn._x_addr_width),
                add_y_addr_width=str(self.add_deisgn._y_addr_width),
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
                x_addr_width=str(self.depthconv1d_0_deisgn._x_addr_width),
                y_addr_width=str(self.add_deisgn._y_addr_width),
                in_channels=str(self._depthconv1d_0.in_channels),
                out_channels=str(self._add.num_dimensions),
                seq_len=str(self._depthconv1d_0.seq_len),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )
