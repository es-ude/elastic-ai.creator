from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class FFN(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        fc1relu: object,
        fc2: object,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._fc1relu = fc1relu
        self._fc2 = fc2

        self._work_library_name = work_library_name

        self.fc1relu_design = self._fc1relu.create_design(name=self._fc1relu.name)
        self.fc2_design = self._fc2.create_design(name=self._fc2.name)

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self.fc1relu_design._x_count,
            y_count=self.fc2_design._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self.fc1relu_design.save_to(destination.create_subpath(self._fc1relu.name))
        self.fc2_design.save_to(destination.create_subpath(self._fc2.name))

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="ffn.tpl.vhd",
            parameters=dict(
                name=self.name,
                fc1relu_name=self._fc1relu.name,
                fc2_name=self._fc2.name,
                data_width=str(self._data_width),
                x_addr_width=str(self.fc1relu_design._x_addr_width),
                y_addr_width=str(self.fc2_design._y_addr_width),
                num_dimensions=str(self.fc1relu_design._num_dimensions),
                fc1_in_features=str(self.fc1relu_design._in_features),
                fc1_out_features=str(self.fc1relu_design._out_features),
                fc2_out_features=str(self.fc2_design._out_features),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="ffn_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self.fc1relu_design._x_addr_width),
                y_addr_width=str(self.fc2_design._y_addr_width),
                num_dimensions=str(self.fc1relu_design._num_dimensions),
                in_features=str(self.fc1relu_design._in_features),
                out_features=str(self.fc1relu_design._out_features),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)
