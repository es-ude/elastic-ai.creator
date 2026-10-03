from collections.abc import Callable

from elasticai.creator.arithmetic import FxpConverter, FxpParams
from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port
from elasticai.creator_plugins.act_func.utils import load_and_plugin


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

    def _compute_io_pairs(self) -> list[tuple[int, int]]:
        ascending_unique_inputs = sorted(set(self._inputs))
        io_pairs = []
        for input_value in ascending_unique_inputs:
            output_value = self._function(input_value)
            io_pairs.append((input_value, output_value))
        return io_pairs

    @property
    def port(self) -> Port:
        return create_port(x_width=self._input_width, y_width=self._output_width)

    def save_to(self, destination: Path, take_vhdl: bool = True) -> None:
        def _save_to_vhdl(destination: Path) -> None:
            process_content = []

            pairs = self._compute_io_pairs()
            for input_value, output_value in pairs[1:-1]:
                process_content.append(
                    f"elsif sx <= to_signed({input_value}, BITWIDTH_INPUT) then "
                    f"return to_signed({output_value}, BITWIDTH_OUTPUT);"
                )
            _, output = pairs[-1]
            process_content.append(f"else return to_signed({output}, BITWIDTH_OUTPUT);")

            self._template = InProjectTemplate(
                file_name="precomputed_scalar_function.tpl.vhd",
                package=self._template_package,
                parameters=dict(
                    name=self.name,
                    input_data_width=str(self._input_width),
                    output_data_width=str(self._output_width),
                ),
            )
            self._template.parameters.update(process_content=process_content)
            destination.create_subpath(self.name).as_file(".vhd").write(self._template)

        def _save_to_verilog(destination: Path) -> None:
            cnv = FxpConverter(
                FxpParams(total_bits=self._input_width, frac_bits=0, signed=True)
            )
            outputs = [val for _, val in self._compute_io_pairs()]
            inputs = [val for val, _ in self._compute_io_pairs()]

            load_and_plugin(
                type="precomputed_scalar",
                id=self.name,
                params={
                    "BITWIDTH_IN": self._input_width,
                    "BITWIDTH_OUT": self._output_width,
                    "NUM_VALUES": len(inputs),
                    "VALUES_X": cnv.integer_to_decimal_string_array_verilog(
                        inputs[::-1]
                    ),
                    "VALUES_Y": cnv.integer_to_decimal_string_array_verilog(
                        outputs[::-1]
                    ),
                },
                packages=["act_func"],
                path2save=str(destination),
            )

        if take_vhdl:
            _save_to_vhdl(destination)
        else:
            _save_to_verilog(destination)
