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


class ResidualBlock(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
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
        self._work_library_name = work_library_name

        self._conv1dbn_0 = conv1dbn_0
        self._conv1dbn_0_relu = conv1dbn_0_relu
        self._conv1dbn_1 = conv1dbn_1
        self._shortcut = shortcut
        self._add = add
        self._relu = relu

        self.conv1dbn_0_deisgn = self._conv1dbn_0.create_design(
            name=self._conv1dbn_0.name
        )
        self.conv1dbn_0_relu_deisgn = self._conv1dbn_0_relu.create_design(
            name=self._conv1dbn_0_relu.name
        )
        self.conv1dbn_1_deisgn = self._conv1dbn_1.create_design(
            name=self._conv1dbn_1.name
        )
        if len(self._shortcut) > 0:
            i = 0
            self.shortcut_design = [None] * len(self._shortcut)
            for submodule in self._shortcut:
                self.shortcut_design[i] = submodule.create_design(name=submodule.name)

        self.add_design = self._add.create_design(name=self._add.name)
        self.relu_design = self._relu.create_design(name=self._relu.name)

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self.conv1dbn_0_deisgn._x_count,
            y_count=self.add_design._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self.conv1dbn_0_deisgn.save_to(
            destination.create_subpath(self._conv1dbn_0.name)
        )
        self.conv1dbn_0_relu_deisgn.save_to(
            destination.create_subpath(self._conv1dbn_0_relu.name)
        )
        self.conv1dbn_1_deisgn.save_to(
            destination.create_subpath(self._conv1dbn_1.name)
        )
        if len(self._shortcut) > 0:
            for i, submodule in enumerate(self._shortcut):
                self.shortcut_design[i].save_to(
                    destination.create_subpath(submodule.name)
                )

        self.add_design.save_to(destination.create_subpath(self._add.name))
        self.relu_design.save_to(destination.create_subpath(self._relu.name))

        template_parameters = dict(
            name=self.name,
            data_width=str(self._data_width),
            conv1dbn_0_x_addr_width=str(self.conv1dbn_0_deisgn._x_addr_width),
            conv1dbn_0_y_addr_width=str(self.conv1dbn_0_deisgn._y_addr_width),
            conv1dbn_1_x_addr_width=str(self.conv1dbn_1_deisgn._x_addr_width),
            conv1dbn_1_y_addr_width=str(self.conv1dbn_1_deisgn._y_addr_width),
            add_x_addr_width=str(self.add_design._x_addr_width),
            add_y_addr_width=str(self.add_design._y_addr_width),
            work_library_name=self._work_library_name,
        )
        template_file_name = "residualblock_no_shortcut.tpl.vhd"

        if hasattr(self, "shortcut_design"):
            template_parameters["shortcut_x_addr_width"] = str(
                self.shortcut_design[0]._x_addr_width
            )
            template_parameters["shortcut_y_addr_width"] = str(
                self.shortcut_design[-1]._y_addr_width
            )
            template_file_name = "residualblock.tpl.vhd"

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=template_file_name,
            parameters=template_parameters,
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="residualblock_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self.conv1dbn_0_deisgn._x_addr_width),
                y_addr_width=str(self.add_design._y_addr_width),
                in_channels=str(self._conv1dbn_0.in_channels),
                out_channels=str(self._add.num_dimensions),
                seq_len=str(self._conv1dbn_0.seq_len),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )
