from typing import Any, Protocol

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.file_generation.savable import Path
from elasticai.creator.vhdl.ghdl_simulation import GHDLSimulator


class Design(Protocol):
    def save_to(self, destination: Path): ...


class Layer(Design, Protocol):
    @property
    def name(self) -> str: ...

    def create_testbench(self, name: str) -> "TestBench": ...


class TestBench(Design, Protocol):
    def set_inputs(self, *inputs) -> None: ...

    def parse_reported_content(self, content: Any) -> Any: ...


class SimulationConstructor(Protocol):
    def __call__(self, workdir: str, top_design_name: str) -> GHDLSimulator: ...


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
        root = OnDiskPath("main", parent=self._working_dir)
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
