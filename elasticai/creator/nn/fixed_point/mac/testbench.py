from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import InProjectTemplate

from ._signal_number_converter import SignalNumberConverter


class InputsFile:
    def __init__(self, inputs):
        self._inputs = inputs
        self._heading = ("reset", "next_sample", "x1", "x2")

    @property
    def content(self):
        return ["$heading_row", "$values"]

    @property
    def parameters(self):
        return {
            "heading_row": ",".join(self._heading),
            "values": [",".join(row) for row in self._inputs],
        }


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

    def __init__(self, total_bits, frac_bits, inputs, name):
        self._number_converter = SignalNumberConverter(
            total_bits=total_bits, frac_bits=frac_bits
        )
        self._inputs = inputs
        self._destination = None
        self._name = name

    def save_to(self, destination: Path):
        self._destination = destination
        test_bench = InProjectTemplate(
            package="elasticai.creator.nn.fixed_point.mac",
            file_name="testbench.tpl.vhd",
            parameters=self._inputs,
        )
        destination.create_subpath(self._name).as_file(".vhd").write(test_bench)
