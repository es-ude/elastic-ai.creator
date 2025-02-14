from collections.abc import Callable

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class LUT(Design):
    def __init__(
        self,
        name: str,
        x_data_width: int,
        y_data_width: int,
        function: Callable[[int], int],
        inputs: list[int],
    ) -> None:
        super().__init__(name=name)

        self._x_data_width = x_data_width
        self._y_data_width = y_data_width

        self._function = function
        self._inputs = inputs

        self._x_count = 2**x_data_width
        self._y_count = 2**y_data_width

        self._x_addr_width = calculate_address_width(self._x_count)
        self._y_addr_width = calculate_address_width(self._y_count)

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._x_data_width,
            y_width=self._y_data_width,
            x_count=self._x_count,
            y_count=self._y_count,
        )

    def _compute_io_pairs(self) -> dict[int, int]:
        inputs_in_descending_order = sorted(self._inputs, reverse=True)
        pairs = dict()
        for number in inputs_in_descending_order:
            pairs[number] = self._function(number)
        return pairs

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="LUT.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_data_width=str(self._x_data_width),
                y_data_width=str(self._y_data_width),
            ),
        )

        process_content = []

        pairs = list(self._compute_io_pairs().items())
        input_value, output_value = pairs[0]

        process_content.append(
            f"if signed_x = {input_value} then "
            f"signed_y <= to_signed({output_value}, {self._y_data_width});"
        )
        for input_value, output_value in pairs[1:-1]:
            process_content.append(
                f"elsif signed_x = {input_value} then "
                f"signed_y <= to_signed({output_value}, {self._y_data_width});"
            )
        _, output = pairs[-1]
        process_content.append(
            f"else signed_y <= to_signed({output}, {self._y_data_width});"
        )
        process_content.append("end if;")

        template.parameters.update(process_content=process_content)

        destination.create_subpath(self.name).as_file(".vhd").write(template)
