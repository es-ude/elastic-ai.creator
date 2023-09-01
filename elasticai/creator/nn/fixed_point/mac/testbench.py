from creator.file_generation.savable import Path
from creator.file_generation.template import InProjectTemplate
from fixed_point.mac._signal_number_converter import SignalNumberConverter


class InputsFile:
    def __init__(self, inputs):
        self._inputs = inputs
        self._heading = ("reset", "next_sample", "x1", "x2")

    @property
    def content(self):
        return ["$heading_row", "$values"]

    @property
    def parameters(self):
        return {"heading": " ".join(self._heading), "values": " ".join(self._inputs)}


class TestBench:
    def __init__(self, total_bits, frac_bits, inputs, name):
        self._number_converter = SignalNumberConverter(
            total_bits=total_bits, frac_bits=frac_bits
        )
        self._inputs = inputs
        self._destination = None
        self._name = name

    def save_to(self, destination: Path):
        self._destination = destination
        inputs_file = InputsFile(self._number_converter.to_signals(self._inputs))
        self._destination.create_subpath("inputs").as_file(".csv").write(inputs_file)
        test_bench = InProjectTemplate(
            package="elasticai.creator.nn.fixed_point.mac",
            file_name="testbench.tpl.vhd",
            parameters={"input_file": f"inputs.txt", "output_file": f"outputs.txt"},
        )
        destination.create_subpath(self._name).as_file(".vhd").write(test_bench)
