"""
1. create testbench
2. save testbench to build folder
3. hand sequence of vhdl files in correct order to simulation tool
4. hand input data for testcase to simulation tool
5. let simulation tool compile files (if necessary)
6. let simulation tool run the simulation
7. parse/deserialize simulation output to required data
"""
import csv
import pathlib
from typing import Any

import pytest
import torch

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.vhdl.ghdl_simulation import GHDLSimulator

from .layer import Conv1d


class SimulatedLayer:
    def __init__(self, testbench, simulator_constructor, working_dir):
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
            print()
            print(inputs)
            header = [x for x in inputs[0].keys()]
            writer = csv.DictWriter(
                f,
                fieldnames=header,
                lineterminator="\n",
                delimiter=" ",
            )
            writer.writeheader()
            writer.writerows(inputs)


def create_ones_conv1d_input_list(
    batch_size: int, in_channels: int, signal_length: int
):
    return [[[1.0] * signal_length] * in_channels] * batch_size


@pytest.mark.simulation
@pytest.mark.parametrize("x", ([[[0.0, 1.0, 1.0]]], [[[1.0, 1.0, 1.0]]], [[[2.0, 1.0, 0.0]]]))
def test_verify_hw_sw_equivalence_3_inputs(x):
    input_data = torch.Tensor(x)
    sw_conv = Conv1d(
        total_bits=4,
        frac_bits=1,
        in_channels=1,
        out_channels=1,
        signal_length=3,
        kernel_size=2,
        bias=True,
    )
    sw_conv.weight.data = torch.ones_like(sw_conv.weight)
    sw_conv.bias.data = torch.ones_like(sw_conv.bias)
    sw_output = sw_conv(input_data)
    design = sw_conv.create_design("conv1d")
    testbench = sw_conv.create_testbench("conv1d_testbench", design)
    build_dir = OnDiskPath("build")
    design.save_to(build_dir.create_subpath("srcs"))
    testbench.save_to(build_dir.create_subpath("testbenches"))
    sim_layer = SimulatedLayer(testbench, GHDLSimulator, working_dir="build")
    sim_output = sim_layer(input_data)
    assert sw_output.tolist() == sim_output


@pytest.mark.simulation
@pytest.mark.parametrize(
    "x",
    (
        create_ones_conv1d_input_list(1, 2, 4),
        [[[0.5, 0.25, -1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]]],
        [[[0.0, 1.0, 1.0, 0.0], [-1.0, 1.0, -1.0, 1.0]]],
    ),
)
def test_verify_hw_sw_equivalence_4_inputs(x):
    input_data = torch.Tensor(x)
    sw_conv = Conv1d(
        total_bits=5,
        frac_bits=2,
        in_channels=2,
        out_channels=2,
        signal_length=4,
        kernel_size=2,
        bias=True,
    )
    sw_conv.weight.data = torch.ones_like(sw_conv.weight)
    sw_conv.bias.data = torch.ones_like(sw_conv.bias)
    sw_output = sw_conv(input_data)
    design = sw_conv.create_design("conv1d")
    testbench = sw_conv.create_testbench("conv1d_testbench", design)
    build_dir = OnDiskPath("build")
    design.save_to(build_dir.create_subpath("srcs"))
    testbench.save_to(build_dir.create_subpath("testbenches"))
    sim_layer = SimulatedLayer(testbench, GHDLSimulator, working_dir="build")
    sim_output = sim_layer(input_data)
    assert sw_output.tolist() == sim_output
