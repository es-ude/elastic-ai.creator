from elasticai.creator.arithmetic import FxpConverter
from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import InProjectTemplate


class MacTestBench:
    def __init__(self, uut, fxp_params, name, uut_name):
        self._fxp_params = fxp_params
        self._converter = FxpConverter(self._fxp_params)
        self._uut = uut
        self._uut_name = uut_name
        self._inputs = None
        self._destination = None
        self.name = name

    def set_inputs(self, *inputs):
        self._inputs = inputs

    def parse_reported_content(self, outputs: list[str]):
        return self._converter.binary_to_rational(outputs[0])

    @property
    def _vector_width(self) -> str:
        return str(len(self._inputs[0]))

    @property
    def _total_bits(self) -> str:
        return str(self._fxp_params.total_bits)

    @property
    def _frac_bits(self) -> str:
        return str(self._fxp_params.frac_bits)

    def save_to(self, destination: Path):
        self._destination = destination
        inputs = self._prepare_inputs_for_test_bench(self._inputs)
        test_bench = InProjectTemplate(
            package="elasticai.creator.nn.fixed_point.mac",
            file_name="testbench.tpl.vhd",
            parameters=inputs
            | {
                "total_width": self._total_bits,
                "vector_width": str(self._vector_width),
                "uut_name": self._uut_name,
            },
        )
        self._uut.save_to(destination)
        destination.create_subpath(self.name).as_file(".vhd").write(test_bench)

    def _prepare_inputs_for_test_bench(self, inputs):
        x1, x2 = inputs
        x1 = map(self._converter.rational_to_binary_string_vhdl, x1)
        x2 = map(self._converter.rational_to_binary_string_vhdl, x2)
        inputs = {
            "x1": ", ".join([f'b"{x}"' for x in x1]),
            "x2": ", ".join([f'b"{x}"' for x in x2]),
        }
        return inputs
