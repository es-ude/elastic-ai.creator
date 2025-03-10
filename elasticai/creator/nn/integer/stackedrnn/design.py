from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class StackedRNN(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        rnn_layers: object,
        num_layers: int,
        work_library_name: str,
    ):
        super().__init__(name=name)

        self._data_width = data_width
        self._work_library_name = work_library_name

        self._rnn_layers = rnn_layers
        self._num_layers = num_layers

        # TODO: only support 1 rnn layer now
        # for i in range(self._num_layers):
        #     self.rnn_layers_design = [None] * self._num_layers
        #     self.rnn_layers_design[i] = self._rnn_layers[i].create_design(
        #         name=self._rnn_layers[i].name
        #     )
        assert self._num_layers == 1, "Only support 1 rnn layer now"
        self.rnn_layer_design = self._rnn_layers[0].create_design(
            name=self._rnn_layers[0].name
        )

        self._x_count = self.rnn_layer_design._x_count
        self._y_count = self.rnn_layer_design._y_count

        self.x_addr_width = self.rnn_layer_design._x_addr_width
        self.y_addr_width = self.rnn_layer_design._y_addr_width

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self._x_count,
            y_count=self._y_count,
        )

    def save_to(self, destination: Path) -> None:
        # for i in range(self._num_layers):
        #     self.rnn_layers_design[i].save_to(destination)

        self.rnn_layer_design.save_to(
            destination.create_subpath(self.rnn_layer_design.name)
        )

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="stackedrnn.tpl.vhd",
            parameters=dict(  # TODO: problems here
                name=self.name,
                x_addr_width=self.x_addr_width,
                y_addr_width=self.y_addr_width,
                x_count=self._x_count,
                y_count=self._y_count,
                layer_name=self.rnn_layer_design.name,
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        test_template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="stackedrnn_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_addr_width=self.x_addr_width,
                y_addr_width=self.y_addr_width,
                x_count=self._x_count,
                y_count=self._y_count,
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(test_template)
