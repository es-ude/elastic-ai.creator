import pytest
import torch

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.vhdl.ghdl_simulation import GHDLSimulator
from elasticai.creator.vhdl.simulated_layer import SimulatedLayer

from .design import MacDesign
from .layer import MacLayer

sw_function_test_data = [
    [[[1, 1, 1], [-1, 1, -1]], [[1, 1, 1], [-1, 1, 1]]],
    [[[1, -1, -1, 1, 1], [1, -1, 1, -1, 1]]],
]


@pytest.mark.parametrize(
    "data",
    (
        ([[[1, 1, 1], [-1, 1, -1]], [[1, 1, 1], [-1, 1, 1]]], [-1, 1]),
        ([[[1, -1, -1, 1, 1], [1, -1, 1, -1, 1]]], [1]),
    ),
)
def test_sw_function(data):
    x, expected = data
    for i, (x1, x2) in enumerate(x):
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        mac = MacLayer(vector_width=x1.shape[0])
        y = mac(x1, x2).item()
        assert y == expected[i]


@pytest.mark.skip(
    reason=(
        "Functionality could be fine, but testbenches is designed for input after input"
        " like 'normal' mac operator"
    )
)
@pytest.mark.simulation
@pytest.mark.parametrize(
    "x",
    (
        # 00010 * 00010 -> 00000 00100 -> 000(00 001)00 -> 00001
        [
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, -1]],
        ],
        # 00010 * 01000 -> 00000 10000 -> 000(00 100)00 -> 00100
        [
            [[-1, 1], [-1, -1]],
        ],
        [[[-1, -1, 1], [1, 1, 1]]],
    ),
)
def test_mac_hw_for_integers(x):
    sw_mac = MacLayer(vector_width=1)
    design: MacDesign = sw_mac.create_design("mac")
    testbench = sw_mac.create_testbench("mac_testbench", design)
    build_dir = OnDiskPath("build")
    design.save_to(build_dir.create_subpath("srcs"))
    testbench.save_to(build_dir.create_subpath("testbenches"))
    sim = SimulatedLayer(testbench, GHDLSimulator, working_dir="build")
    print(f"{x=}")
    actual = sim(x)
    for i, (x1, x2) in enumerate(x):
        y = sw_mac(
            torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)
        )
        assert y == actual[i]
