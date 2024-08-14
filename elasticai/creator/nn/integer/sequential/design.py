from typing import cast

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.nn.integer.linear.design import Linear
from elasticai.creator.nn.sequential.design import Sequential as _SequentialDesign
from elasticai.creator.vhdl.design.design import Design


class Sequential(_SequentialDesign):
    def __init__(self, sub_designs: list[Design], *, name: str) -> None:
        super().__init__(sub_designs, name=name)

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
                layer_name=self.name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(network_template)

        in_features = cast(Linear, self._subdesigns[0])._in_features
        out_features = cast(Linear, self._subdesigns[-1])._out_features

        network_template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="network_tb.tpl.vhd",
            parameters=dict(
                x_address_width=str(self._x_address_width),
                y_address_width=str(self._y_address_width),
                data_width=str(self._x_width),
                in_features=str(in_features),
                out_features=str(out_features),
                name=self.name,
            ),
        )
        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            network_template_test
        )
