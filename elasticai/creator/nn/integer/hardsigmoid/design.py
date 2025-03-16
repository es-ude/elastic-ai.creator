from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class HardSigmoid(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        quantized_three: int,
        quantized_minus_three: int,
        quantized_one: int,
        quantized_zero: int,
        tmp: int,  # Note: at the moment hardware side only supports 16 bits signed of TEMP.
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._work_library_name = work_library_name

        self._quantized_three = quantized_three
        self._quantized_minus_three = quantized_minus_three
        self._quantized_one = quantized_one
        self._quantized_zero = quantized_zero
        self._tmp = tmp

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
        )

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="hardsigmoid.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                three_threshold=str(self._quantized_three),
                minus_three_threshold=str(self._quantized_minus_three),
                zero_output=str(self._quantized_zero),
                one_output=str(self._quantized_one),
                tmp_threshold=str(self._tmp),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="hardsigmoid_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )
