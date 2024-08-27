from pathlib import Path
from typing import Any, Protocol

from elasticai.creator.vhdl.ghdl_simulation import GHDLSimulator


class Design(Protocol):
    name: str

    def save_to(self, destination: Path):
        ...


class TestBench(Design, Protocol):
    def set_inputs(self, *inputs) -> None:
        ...

    def parse_reported_content(self, content: list[str]) -> Any:
        ...


class Layer(Protocol):
    @property
    def name(self) -> str:
        ...

    def create_testbench(self, name: str) -> TestBench:
        ...


class SimulationConstructor(Protocol):
    def __call__(self, workdir: str, top_design_name: str) -> GHDLSimulator:
        ...


class SimulatedLayer:
    def __init__(
        self,
        layer_under_test: Layer,
        simulation_constructor: SimulationConstructor,
        working_dir: str,
    ):
        self._layer_under_test = layer_under_test
        self._simulation_constructor = simulation_constructor
        self._working_dir = working_dir
        self._inputs = None

    def __call__(self, *inputs):
        root = Path(self._working_dir) / "main"
        testbench_name = f"testbench_{self._layer_under_test.name}"
        testbench = self._layer_under_test.create_testbench(testbench_name)
        testbench.set_inputs(*inputs)
        testbench.save_to(root)
        runner = self._simulation_constructor(
            workdir=f"{self._working_dir}", top_design_name=testbench.name
        )
        runner.initialize()
        runner.run()
        actual = testbench.parse_reported_content(runner.getReportedContent())
        return actual
