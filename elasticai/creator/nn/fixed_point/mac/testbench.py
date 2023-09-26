from functools import partial

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import InProjectTemplate

from .number_conversion import bits_to_rational, convert_rational_to_bit_pattern


class TestBench:
    """
    The testbench aims to provide an interface between hw and sw engineer for
    generating hw/sw tests for a hw/sw module pair.
    The use case is as follows:
     - Given a translatable software module and some input data `X`, we
        - call the software module with the provided data and record the outputs
        - generate the hw design (unit under test UUT) for the sw equivalent
        - generate a testbench, that feeds the same input data `X` to the UUT and writes the output data to a file or stdout
        - compare the output observed by the testbench to the output of the sw module
    """

    def __init__(self, total_bits, frac_bits, x1, x2, name, vector_width):
        self._total_bits = total_bits
        self._frac_bits = frac_bits
        self._vector_width = vector_width

        def prepare_inputs_for_test_bench(x1, x2):
            to_bit_pattern = partial(
                convert_rational_to_bit_pattern,
                total_bits=self._total_bits,
                frac_bits=self._frac_bits,
            )
            x1 = map(to_bit_pattern, x1)
            x2 = map(to_bit_pattern, x2)
            inputs = {
                "x1": ", ".join([f'b"{x}"' for x in x1]),
                "x2": ", ".join([f'b"{x}"' for x in x2]),
            }
            return inputs

        self._inputs = prepare_inputs_for_test_bench(x1, x2)
        self._destination = None
        self.name = name

    def parse_reported_content(self, outputs: list[str]):
        return bits_to_rational(outputs[0], frac_bits=self._frac_bits)

    def save_to(self, destination: Path):
        self._destination = destination
        test_bench = InProjectTemplate(
            package="elasticai.creator.nn.fixed_point.mac",
            file_name="testbench.tpl.vhd",
            parameters=self._inputs
            | {
                "total_width": str(self._total_bits),
                "frac_width": str(self._frac_bits),
                "vector_width": str(self._vector_width),
            },
        )
        destination.create_subpath(self.name).as_file(".vhd").write(test_bench)
