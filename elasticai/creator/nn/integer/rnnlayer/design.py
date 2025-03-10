from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class RNNLayer(Design):
    def __init__(
        self,
        name: str,
        cell_type: str,
        data_width: int,
        rnn_cell: object,
        work_library_name: str,
    ):
        super().__init__(name=name)

        self._data_width = data_width
        self._work_library_name = work_library_name

        self._cell_type = cell_type
        self._rnn_cell = rnn_cell
        self.rnn_cell_deisgn = self._rnn_cell.create_design(name=self._rnn_cell.name)

        self._x_1_count = self.rnn_cell_deisgn._x_1_count
        self._x_2_count = self.rnn_cell_deisgn._x_2_count
        self._y_1_count = (
            self._rnn_cell.batch_size
            * self._rnn_cell.window_size
            * self._rnn_cell.hidden_size
        )
        self._y_2_count = self.rnn_cell_deisgn._y_2_count

        self._x_1_addr_width = self.rnn_cell_deisgn._x_1_addr_width
        self._x_2_addr_width = self.rnn_cell_deisgn._x_2_addr_width
        self._y_1_addr_width = calculate_address_width(self._y_3_count)
        self._y_2_addr_width = self.rnn_cell_deisgn._y_2_addr_width

        if self._cell_type == "lstm":
            self._x_3_count = self.rnn_cell_deisgn._x_3_count
            self._x_3_addr_width = self.rnn_cell_deisgn._x_3_addr_width
            self._y_3_count = self.rnn_cell_deisgn._y_3_count
            self._y_3_addr_width = self.rnn_cell_deisgn._y_3_addr_width

    @property
    def port(self) -> Port:
        return create_port(
            x_1_width=self._data_width,
            x_2_width=self._data_width,
            x_3_width=self._data_width,
            y_1_width=self._data_width,
            y_2_width=self._data_width,
            y_3_width=self._data_width,
            x_1_count=self._x_1_count,
            x_2_count=self._x_2_count,
            x_3_count=self._x_3_count,
            y_1_count=self._y_1_count,
            y_2_count=self._y_2_count,
            y_3_count=self._y_3_count,
        )

    def save_to(self, destination: Path) -> None:
        self.rnn_cell_deisgn.save_to(destination)

        template_params = {
            "name": self.name,
            "data_width": self._data_width,
            "x_1_count": self._x_1_count,
            "x_2_count": self._x_2_count,
            "y_1_count": self._y_1_count,
            "y_2_count": self._y_2_count,
            "x_1_addr_width": self._x_1_addr_width,
            "x_2_addr_width": self._x_2_addr_width,
            "y_1_addr_width": self._y_1_addr_width,
            "y_2_addr_width": self._y_2_addr_width,
            "rnn_cell_name": self.rnn_cell_deisgn.name,
            "work_library_name": self._work_library_name,
        }
        if self._cell_type == "lstm":
            template_params["x_3_count"] = self._x_3_count
            template_params["y_3_count"] = self._y_3_count
            template_params["x_3_addr_width"] = self._x_3_addr_width
            template_params["y_3_addr_width"] = self._y_3_addr_width

        file_name = f"{self._cell_type}layer.tpl.vhd"
        tb_file_name = f"{self._cell_type}layer_tb.tpl.vhd"

        test_template_params = template_params.copy()

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=file_name,
            parameters=template_params,
            cell_name=self._rnn_cell.name,
            work_library_name=self._work_library_name,
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        test_template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=tb_file_name,
            parameters=test_template_params,
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(test_template)
