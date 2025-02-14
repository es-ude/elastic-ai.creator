from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class Encoder(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        encoder_layers: object,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._encoder_layers = encoder_layers

        self._work_library_name = work_library_name

        self.encoder_layer_1_design = self._encoder_layers[0].create_design(
            name=self._encoder_layers[0].name
        )

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self.encoder_layer_1_design._x_count,
            y_count=self.encoder_layer_1_design._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self.encoder_layer_1_design.save_to(destination)

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="encoder.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self.encoder_layer_1_design._x_addr_width),
                y_addr_width=str(self.encoder_layer_1_design._y_addr_width),
                num_dimensions=str(self.encoder_layer_1_design._num_dimensions),
                in_features=str(self.encoder_layer_1_design._in_features),
                out_features=str(self.encoder_layer_1_design._out_features),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="encoder_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self.encoder_layer_1_design._x_addr_width),
                y_addr_width=str(self.encoder_layer_1_design._y_addr_width),
                num_dimensions=str(self.encoder_layer_1_design._num_dimensions),
                in_features=str(self.encoder_layer_1_design._in_features),
                out_features=str(self.encoder_layer_1_design._out_features),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)
