from collections.abc import Callable

from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.design_base.design import Design
from elasticai.creator.vhdl.design_base.ports import Port
from elasticai.creator.vhdl.savable import Path


class PrecomputedMonotonicIncreasingScalarFunction(Design):
    _template_package = module_to_package(__name__)

    def __init__(
        self,
        name: str,
        width: int,
        function: Callable[[int], int],
        inputs: list[int],
    ) -> None:
        super().__init__(name)
        self._width = width
        self._function = function
        self._inputs = inputs
        self._template = InProjectTemplate(
            file_name="precomputed_monotonic_increasing_scalar_function.tpl.vhd",
            package=self._template_package,
            parameters=dict(name=self.name, data_width=str(width)),
        )

    def _compute_io_pairs(self) -> dict[int, int]:
        inputs_in_descending_order = sorted(self._inputs, reverse=True)
        pairs = dict()
        for number in inputs_in_descending_order:
            pairs[number] = self._function(number)
        return pairs

    @property
    def port(self) -> Port:
        return create_port(x_width=self._width, y_width=self._width)

    def save_to(self, destination: Path) -> None:
        process_content = []

        pairs = list(self._compute_io_pairs().items())
        input_value, output_value = pairs[0]
        process_content.append(
            f"if signed_x <= {input_value} then "
            f"signed_y <= to_signed({output_value}, {self._width});"
        )
        for input_value, output_value in pairs[1:-1]:
            process_content.append(
                f"elsif signed_x <= {input_value} then "
                f"signed_y <= to_signed({output_value}, {self._width});"
            )
        _, output = pairs[-1]
        process_content.append(f"else signed_y <= to_signed({output}, {self._width});")
        process_content.append("end if;")

        self._template.parameters.update(process_content=process_content)
        destination.create_subpath(self.name).as_file(".vhd").write(self._template)
