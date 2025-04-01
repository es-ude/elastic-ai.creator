import csv
import pathlib
from typing import Any

from elasticai.creator.file_generation.savable import Path


class Testbench:
    @property
    def name(self) -> str:
        raise NotImplementedError

    def save_to(self, destination: Path) -> None:
        raise NotImplementedError

    def prepare_inputs(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def parse_reported_content(self, *args, **kwargs: Any) -> Any:
        raise NotImplementedError


class SimulatedLayer:
    def __init__(self, testbench: Testbench, simulator_constructor, working_dir):
        self._testbench = testbench
        self._simulator_constructor = simulator_constructor
        self._working_dir = working_dir
        self._inputs_file_path = (
            f"{self._working_dir}/{self._testbench.name}_inputs.csv"
        )

    def __call__(self, inputs: Any) -> Any:
        runner = self._simulator_constructor(
            workdir=f"{self._working_dir}", top_design_name=self._testbench.name
        )
        inputs = self._testbench.prepare_inputs(inputs)
        self._write_csv(inputs)
        runner.add_generic(
            INPUTS_FILE_PATH=str(pathlib.Path(self._inputs_file_path).absolute())
        )
        runner.initialize()
        runner.run()
        actual = self._testbench.parse_reported_content(runner.getReportedContent())
        return actual

    def _write_csv(self, inputs):
        with open(self._inputs_file_path, "w") as f:
            header = [x for x in inputs[0].keys()]
            writer = csv.DictWriter(
                f,
                fieldnames=header,
                lineterminator="\n",
                delimiter=" ",
            )
            writer.writeheader()
            writer.writerows(inputs)
