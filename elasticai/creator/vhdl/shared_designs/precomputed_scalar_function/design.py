from collections.abc import Callable

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class PrecomputedScalarFunction(Design):
    _template_package = module_to_package(__name__)

    def __init__(
        self,
        name: str,
        input_width: int,
        output_width: int,
        function: Callable[[int], int],
        inputs: list[int],
    ) -> None:
        super().__init__(name)
        self._input_width = input_width
        self._output_width = output_width
        self._function = function
        self._inputs = inputs
        self._template = InProjectTemplate(
            file_name="precomputed_scalar_function.tpl.vhd",
            package=self._template_package,
            parameters=dict(
                name=self.name,
                input_data_width=str(self._input_width),
                output_data_width=str(self._output_width),
            ),
        )

    def _compute_io_pairs(self) -> list[tuple[int, int]]:
        ascending_unique_inputs = sorted(set(self._inputs))
        io_pairs = []
        for input_value in ascending_unique_inputs:
            _assert_value_is_representable_with_n_bits(input_value, self._input_width)
            output_value = self._function(input_value)
            _assert_value_is_representable_with_n_bits(output_value, self._output_width)
            io_pairs.append((input_value, output_value))
        return io_pairs

    @property
    def port(self) -> Port:
        return create_port(x_width=self._input_width, y_width=self._output_width)

    def save_to(self, destination: Path) -> None:
        process_content = []

        pairs = self._compute_io_pairs()
        input_value, output_value = pairs[0]
        process_content.append(
            f"if signed_x <= {input_value} then "
            f"signed_y <= to_signed({output_value}, {self._output_width});"
        )
        for input_value, output_value in pairs[1:-1]:
            process_content.append(
                f"elsif signed_x <= {input_value} then "
                f"signed_y <= to_signed({output_value}, {self._output_width});"
            )
        _, output = pairs[-1]
        process_content.append(
            f"else signed_y <= to_signed({output}, {self._output_width});"
        )
        process_content.append("end if;")

        self._template.parameters.update(process_content=process_content)
        destination.create_subpath(self.name).as_file(".vhd").write(self._template)


def _assert_value_is_representable_with_n_bits(value: int, n_bits: int) -> None:
    min_value = 2 ** (n_bits - 1) * (-1)
    max_value = 2 ** (n_bits - 1) - 1

    if value < min_value or value > max_value:
        raise ValueError(
            f"The value '{value}' cannot be represented "
            f"with {n_bits} in two's complement representation."
        )
