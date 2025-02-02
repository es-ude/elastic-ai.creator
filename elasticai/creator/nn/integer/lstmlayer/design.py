from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class LSTMLayer(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        lstm_cell: object,
        work_library_name: str,
    ):
        super().__init__(name=name)

        self._data_width = data_width
        self._work_library_name = work_library_name

        self._lstm_cell = lstm_cell

        self.lstm_cell_deisgn = self._lstm_cell.create_design(name=self._lstm_cell.name)

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self.lstm_cell_deisgn._x_count,
            y_count=self.lstm_cell_deisgn._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self.lstm_cell_deisgn.save_to(destination)

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="lstmlayer.tpl.vhd",
            parameters=dict(
                name=self.name,
                lstm_cell_x_1_addr_width=self.lstm_cell_deisgn._x_1_addr_width,
                lstm_cell_x_2_addr_width=self.lstm_cell_deisgn._x_2_addr_width,
                lstm_cell_x_3_addr_width=self.lstm_cell_deisgn._x_3_addr_width,
                lstm_cell_y_1_addr_width=self.lstm_cell_deisgn._y_1_addr_width,
                lstm_cell_y_2_addr_width=self.lstm_cell_deisgn._y_2_addr_width,
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        test_template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="lstmlayer_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_1_addr_width=self.lstm_cell_deisgn._x_1_addr_width,
                x_2_addr_width=self.lstm_cell_deisgn._x_2_addr_width,
                x_3_addr_width=self.lstm_cell_deisgn._x_3_addr_width,
                y_1_addr_width=self.lstm_cell_deisgn._y_1_addr_width,
                y_2_addr_width=self.lstm_cell_deisgn._y_2_addr_width,
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(test_template)
