from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import InProjectTemplate


class MacTestBench:
    def __init__(self, uut, name, uut_name):
        self._uut = uut
        self._uut_name = uut_name
        self._inputs = None
        self._destination = None
        self.name = name

    def set_inputs(self, *inputs):
        self._inputs = inputs

    def parse_reported_content(self, outputs: list[str]):
        return 2 * int(outputs[0]) - 1

    @property
    def _width(self) -> str:
        return str(len(self._inputs[0]))

    def save_to(self, destination: Path):
        self._destination = destination
        inputs = self._prepare_inputs_for_test_bench(self._inputs)
        test_bench = InProjectTemplate(
            package="elasticai.creator.nn.binary.mac",
            file_name="testbench.tpl.vhd",
            parameters=inputs
            | {
                "uut_name": self._uut_name,
                "name": self.name,
                "total_width": self._width,
            },
        )
        self._uut.save_to(destination)
        destination.create_subpath(self.name).as_file(".vhd").write(test_bench)

    def _prepare_inputs_for_test_bench(self, inputs):
        x1, x2 = inputs

        def zero_one(xs):
            return ["0" if x < 0 else "1" for x in xs]

        def to_string(xs):
            return '"{}"'.format("".join(xs))

        x1 = zero_one(x1)
        x2 = zero_one(x2)
        inputs = {
            "x1": to_string(x1),
            "x2": to_string(x2),
        }
        return inputs
