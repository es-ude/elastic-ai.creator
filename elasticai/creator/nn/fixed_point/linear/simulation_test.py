import pytest
import torch

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.vhdl.ghdl_simulation import GHDLSimulator
from elasticai.creator.vhdl.simulated_layer import SimulatedLayer

from .layer import Linear


def create_ones_input_list(batch_size: int, in_feature_num: int):
    return [[[1.0] * in_feature_num]] * batch_size


@pytest.mark.simulation
@pytest.mark.parametrize(
    "x", ([[[0.0, 1.0, 1.0]]], [[[1.0, 1.0, 1.0]]], [[[2.0, 1.0, 0.0]]])
)
def test_verify_hw_sw_equivalence_3_inputs(x):
    input_data = torch.Tensor(x)
    sw_conv = Linear(
        in_features=3,
        out_features=2,
        total_bits=4,
        frac_bits=1,
        bias=True,
    )
    sw_conv.weight.data = torch.ones_like(sw_conv.weight)
    sw_conv.bias.data = torch.ones_like(sw_conv.bias)
    sw_output = sw_conv(input_data)
    design = sw_conv.create_design("linear")
    testbench = sw_conv.create_testbench("linear_testbench", design)
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
        create_ones_input_list(2, 4),
        [[[0.5, 0.25, -1.0, 1.0]], [[-1.0, 1.0, -1.0, 1.0]]],
        [[[0.0, 1.0, 1.0, 0.0]], [[-1.0, 1.0, -1.0, 1.0]]],
    ),
)
def test_verify_hw_sw_equivalence_4_inputs(x):
    input_data = torch.Tensor(x)
    sw_conv = Linear(
        in_features=4,
        out_features=10,
        total_bits=4,
        frac_bits=1,
        bias=True,
    )
    sw_conv.weight.data = torch.ones_like(sw_conv.weight)
    sw_conv.bias.data = torch.ones_like(sw_conv.bias)
    sw_output = sw_conv(input_data)
    design = sw_conv.create_design("linear")
    print(f"{design.in_feature_num=}")
    print(f"{design.out_feature_num=}")
    testbench = sw_conv.create_testbench("linear_testbench", design)
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
        [
            [[0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0]],
            [[2.0, 0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0, 0.0]],
            [[1.0, 0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0, 1.0]],
            [[3.0, 0.0, 0.0, 0.0]],
            [[1.0, 2.0, 0.0, 0.0]],
            [[1.0, 0.0, 2.0, 0.0]],
            [[1.0, 0.0, 0.0, 2.0]],
            [[1.0, 1.0, 0.0, 0.0]],
            [[0.0, 1.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0, 1.0]],
            [[1.0, 0.0, 0.0, 1.0]],
            [[-1.0, 0.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0, 0.0]],
            [[0.0, 0.0, -1.0, 0.0]],
            [[0.0, 0.0, 0.0, -1.0]],
            [[2.0, 0.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0, 0.0]],
            [[0.0, 0.0, 2.0, 0.0]],
            [[0.0, 0.0, 0.0, 2.0]],
        ],
    ),
)
def test_verify_hw_sw_equivalence_4_inputs_3_outputs(x):
    input_data = torch.Tensor(x)
    sw_conv = Linear(
        in_features=4,
        out_features=3,
        total_bits=8,
        frac_bits=2,
        bias=True,
    )
    sw_conv.weight.data = torch.ones_like(sw_conv.weight) * 2
    sw_conv.bias.data = torch.Tensor([1.0, 2.0, -1.0])
    sw_output = sw_conv(input_data)
    design = sw_conv.create_design("linear")
    testbench = sw_conv.create_testbench("linear_testbench", design)
    build_dir = OnDiskPath("build")
    design.save_to(build_dir.create_subpath("srcs"))
    testbench.save_to(build_dir.create_subpath("testbenches"))
    sim_layer = SimulatedLayer(testbench, GHDLSimulator, working_dir="build")
    sim_output = sim_layer(input_data)
    assert sw_output.tolist() == sim_output
