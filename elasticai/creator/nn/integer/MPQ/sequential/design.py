from typing import cast

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.nn.integer.linear.design import Linear
from elasticai.creator.nn.integer.ram.design import Ram
from elasticai.creator.nn.sequential.design import Sequential as _SequentialDesign
from elasticai.creator.vhdl.design.design import Design


class Sequential(_SequentialDesign):
    def __init__(self, sub_designs: list[Design], *, name: str) -> None:
        super().__init__(sub_designs, name=name)

        self._x_count = self._subdesigns[0]._x_count
        self._y_count = self._subdesigns[-1]._y_count

        self._x_data_width = self._subdesigns[0]._x_data_width
        self._y_data_width = self._subdesigns[-1]._y_data_width

    def save_to(self, destination: Path) -> None:
        self._save_subdesigns(destination)

        destination = destination.create_subpath(self.name)

        network_template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="network.tpl.vhd",
            parameters=dict(
                layer_connections=self._generate_connections_code(),
                layer_instantiations=self._generate_instantiations(),
                signal_definitions=self._generate_signal_definitions(),
                x_address_width=str(self._x_address_width),
                y_address_width=str(self._y_address_width),
                x_width=str(self._x_width),
                y_width=str(self._y_width),
                name=self.name,
                work_library_name="work",
            ),
        )

        destination.create_subpath(self.name).as_file(".vhd").write(network_template)

        network_template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="network_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                work_library_name="work",
                x_address_width=str(self._x_address_width),
                y_address_width=str(self._y_address_width),
                x_data_width=str(self._x_data_width),
                y_data_width=str(self._y_data_width),
                x_count=str(self._x_count),
                y_count=str(self._y_count),
            ),
        )
        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            network_template_test
        )

        ram = Ram(name=self.name + "_ram")
        ram.save_to(destination)
