from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class HardTanh(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        quantized_one: int,
        quantized_minus_one: int,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._work_library_name = work_library_name

        self._quantized_one = quantized_one
        self._quantized_minus_one = quantized_minus_one

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
        )

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="hardtanh.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                one_threshold=str(self._quantized_one),
                minus_one_threshold=str(self._quantized_minus_one),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="hardtanh_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )
