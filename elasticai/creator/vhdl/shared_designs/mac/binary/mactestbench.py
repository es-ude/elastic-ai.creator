from collections import defaultdict

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import InProjectTemplate
from elasticai.creator.vhdl.simulated_layer import Testbench


class MacTestBench(Testbench):
    def __init__(self, uut, name, uut_name):
        self._uut = uut
        self._uut_name = uut_name
        self._inputs = None
        self._destination = None
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def parse_reported_content(self, content: list[str]):
        """
        Somehow there is no content...
        """
        for line in content:
            print(line)

    def save_to(self, destination: Path):
        self._destination = destination
        test_bench = InProjectTemplate(
            package="elasticai.creator.vhdl.shared_designs.mac.binary",
            file_name="testbench.tpl.vhd",
            parameters={
                "uut_name": self._uut_name,
                "name": self.name,
            },
        )
        destination.create_subpath(self.name).as_file(".vhd").write(test_bench)

    def prepare_inputs(self, inputs) -> list[dict]:
        def zero_one(x):
            if x < 0:
                return "0"
            else:
                return "1"

        prepared_inputs = []
        for batch in inputs:
            prepared_inputs.append({})
            for i in range(0, len(batch[0])):
                prepared_inputs[-1].update(
                    {f"x1_{i}": zero_one(batch[0][i]), f"x2_{i}": zero_one(batch[1][i])}
                )
        return prepared_inputs
