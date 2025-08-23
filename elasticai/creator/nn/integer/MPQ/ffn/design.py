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
        fc1: object,
        relu: object,
        fc2: object,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._fc1_name = fc1.name
        self._relu_name = relu.name
        self._fc2_name = fc2.name
        self._work_library_name = work_library_name

        self._fc1_design = fc1.create_design(name=self._fc1_name)
        self._relu_design = relu.create_design(name=self._relu_name)
        self._fc2_design = fc2.create_design(name=self._fc2_name)

        self._x_data_width = self._fc1_design._x_data_width
        self._y_data_width = self._fc2_design._y_data_width

        self._x_addr_width = self._fc1_design._x_addr_width
        self._y_addr_width = self._fc2_design._y_addr_width

        self._x_count = self._fc1_design._x_count
        self._y_count = self._fc2_design._y_count

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._x_data_width,
            y_width=self._y_data_width,
            x_count=self._x_count,
            y_count=self._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self._fc1_design.save_to(destination.create_subpath(self._fc1_name))
        self._relu_design.save_to(destination.create_subpath(self._relu_name))
        self._fc2_design.save_to(destination.create_subpath(self._fc2_name))

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="ffn.tpl.vhd",
            parameters=dict(
                name=self.name,
                fc1_name=self._fc1_name,
                relu_name=self._relu_name,
                fc2_name=self._fc2_name,
                work_library_name=self._work_library_name,
                x_data_width=str(self._x_data_width),
                fc1_y_data_width=str(self._fc1_design._y_data_width),
                relu_x_data_width=str(self._relu_design._x_data_width),
                relu_y_data_width=str(self._relu_design._y_data_width),
                fc2_x_data_width=str(self._fc2_design._x_data_width),
                y_data_width=str(self._y_data_width),
                x_addr_width=str(self._x_addr_width),
                fc1_y_addr_width=str(self._fc1_design._y_data_width),
                relu_y_addr_width=str(self._relu_design._y_data_width),
                y_addr_width=str(self._y_addr_width),
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="ffn_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                work_library_name=self._work_library_name,
                x_data_width=str(self._x_data_width),
                y_data_width=str(self._y_data_width),
                x_addr_width=str(self._fc1_design._x_addr_width),
                y_addr_width=str(self._fc2_design._y_addr_width),
                x_count=str(self._x_count),
                y_count=str(self._y_count),
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)
