from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class StackedLSTM(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        lstm_layers: object,
        num_layers: int,
        work_library_name: str,
    ):
        super().__init__(name=name)

        self._data_width = data_width
        self._work_library_name = work_library_name

        self._lstm_layers = lstm_layers
        self._num_layers = num_layers

        for i in range(self._num_layers):
            self.lstm_layers_design = [None] * self._num_layers
            self.lstm_layers_design[i] = self._lstm_layers[i].create_design(
                name=self._lstm_layers[i].name
            )

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self.lstm_layers_design[0]._x_count,
            y_count=self.lstm_layers_design[-1]._y_count,
        )

    def save_to(self, destination: Path) -> None:
        for i in range(self._num_layers):
            self.lstm_layers_design[i].save_to(destination)

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="stackedlstm.tpl.vhd",
            parameters=dict(  # TODO: problems here
                name=self.name,
                lstm_layers_x_1_addr_width=self.lstm_layers_design[0]._x_1_addr_width,
                lstm_layers_x_2_addr_width=self.lstm_layers_design[0]._x_2_addr_width,
                lstm_layers_x_3_addr_width=self.lstm_layers_design[0]._x_3_addr_width,
                lstm_layers_y_1_addr_width=self.lstm_layers_design[-1]._y_1_addr_width,
                lstm_layers_y_2_addr_width=self.lstm_layers_design[-1]._y_2_addr_width,
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        test_template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="stackedlstm_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_1_addr_width=self.lstm_layers_design[0]._x_1_addr_width,
                x_2_addr_width=self.lstm_layers_design[0]._x_2_addr_width,
                x_3_addr_width=self.lstm_layers_design[0]._x_3_addr_width,
                y_1_addr_width=self.lstm_layers_design[-1]._y_1_addr_width,
                y_2_addr_width=self.lstm_layers_design[-1]._y_2_addr_width,
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(test_template)
