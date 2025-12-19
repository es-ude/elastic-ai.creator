from pathlib import Path

import pytest
import torch

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.fixed_point.conv1d.layer import Conv1d
from elasticai.creator.testing.ghdl_simulation import GHDLSimulator
from elasticai.creator.testing.simulated_layer import SimulatedLayer


def create_ones_conv1d_input_list(
    batch_size: int, in_channels: int, signal_length: int
):
    return [[[1.0] * signal_length] * in_channels] * batch_size


@pytest.mark.simulation
@pytest.mark.parametrize(
    "x", ([[[0.0, 1.0, 1.0]]], [[[1.0, 1.0, 1.0]]], [[[2.0, 1.0, 0.0]]])
)
def test_verify_hw_sw_equivalence_3_inputs(
    x: list[list[list[float]]], tmp_path: Path
) -> None:
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

    build_dir = OnDiskPath(name=tmp_path.name, parent=str(tmp_path.parent))
    design.save_to(build_dir.create_subpath("srcs"))
    testbench.save_to(build_dir.create_subpath("testbenches"))

    sim_layer = SimulatedLayer(testbench, GHDLSimulator, working_dir=tmp_path)
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
def test_verify_hw_sw_equivalence_4_inputs(
    x: list[list[list[float]]], tmp_path: Path
) -> None:
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

    build_dir = OnDiskPath(name=tmp_path.name, parent=str(tmp_path.parent))
    design.save_to(build_dir.create_subpath("srcs"))
    testbench.save_to(build_dir.create_subpath("testbenches"))

    sim_layer = SimulatedLayer(testbench, GHDLSimulator, working_dir=tmp_path)
    sim_output = sim_layer(input_data)

    assert sw_output.tolist() == sim_output
