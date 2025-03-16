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
        inputs_size: int,
        window_size: int,
        hidden_size: int,
        cell_type: str,
        data_width: int,
        rnn_cell: object,
        work_library_name: str,
    ):
        super().__init__(name=name)

        self._inputs_size = inputs_size
        self._window_size = window_size
        self._hidden_size = hidden_size

        self._data_width = data_width
        self._work_library_name = work_library_name

        self._cell_type = cell_type
        self._rnn_cell = rnn_cell
        self.rnn_cell_deisgn = self._rnn_cell.create_design(name=self._rnn_cell.name)

        # q_inputs
        self._x_1_count = self._window_size * self._inputs_size
        # q_h_prev
        self._x_2_count = self._hidden_size

        # q_outputs
        self._y_1_count = self._window_size * self._hidden_size
        # q_h_next
        self._y_2_count = self._hidden_size

        self._x_1_addr_width = calculate_address_width(self._x_1_count)
        self._x_2_addr_width = calculate_address_width(self._x_2_count)
        self._y_1_addr_width = calculate_address_width(self._y_2_count)
        self._y_2_addr_width = calculate_address_width(self._y_2_count)

        if self._cell_type == "lstm":
            # q_c_prev
            self._x_3_count = self._hidden_size
            self._x_3_addr_width = calculate_address_width(self._x_3_count)
            # q_c_next
            self._y_3_count = self._hidden_size
            self._y_3_addr_width = calculate_address_width(self._y_3_count)

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
            "data_width": str(self._data_width),
            "x_1_count": str(self._x_1_count),
            "x_2_count": str(self._x_2_count),
            "y_1_count": str(self._y_1_count),
            "y_2_count": str(self._y_2_count),
            "x_1_addr_width": str(self._x_1_addr_width),
            "x_2_addr_width": str(self._x_2_addr_width),
            "y_1_addr_width": str(self._y_1_addr_width),
            "y_2_addr_width": str(self._y_2_addr_width),
            "cell_name": self._rnn_cell.name,
            "work_library_name": self._work_library_name,
        }
        if self._cell_type == "lstm":
            template_params["x_3_count"] = str(self._x_3_count)
            template_params["y_3_count"] = str(self._y_3_count)
            template_params["x_3_addr_width"] = str(self._x_3_addr_width)
            template_params["y_3_addr_width"] = str(self._y_3_addr_width)

        file_name = f"{self._cell_type}layer.tpl.vhd"
        tb_file_name = f"{self._cell_type}layer_tb.tpl.vhd"

        test_template_params = template_params.copy()

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=file_name,
            parameters=template_params,
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        test_template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=tb_file_name,
            parameters=test_template_params,
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(test_template)
