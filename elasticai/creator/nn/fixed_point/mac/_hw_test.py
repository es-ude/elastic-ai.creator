import pytest
import torch
from fixed_point.number_converter import FXPParams

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.vhdl.ghdl_simulation import GHDLSimulation

from .layer import MacLayer

integer_test_data = [
    (FXPParams(4, 0), x1, x2)
    for x1, x2 in [
        ((1.0, 0.0), (-1.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0)),
        ((1.0, 0.0), (3.0, 0.0)),
        ((2.0, 0.0), (4.0, 0.0)),
        ((-2.0, 0.0), (4.0, 0.0)),
        ((-8.0, -8.0), (7.0, 7.0)),
        ((1.0, 1.0), (1.0, 1.0)),
    ]
]

fractions_test_data = [
    (FXPParams(5, 2), x1, x2)
    for x1, x2 in [
        # 00010 * 00010 -> 00000 00100 -> 000(00 001)00 -> 00001
        ((0.5, 0.0), (0.5, 0.0)),
        # 00010 * 01000 -> 00000 10000 -> 000(00 100)00 -> 00100
        ((0.5, 0.0), (2.0, 0.0)),
        ((0.25, 0.0), (0.5, 0.0)),
        ((0.5, 0.5), (0.5, 0.5)),
        # 00001 * 00010 + 00100 * 00001 -> 00000 00010 + 00000 00100 -> 00000 0110 -> 00(000 01)10 -> 00(000 10)00 -> 00010
        ((0.25, 1.0), (0.5, 0.25)),
        ((-0.25, -1.0), (0.5, 0.5)),
        ((-4.0, -4.0), (3.75, 3.75)),
    ]
]


@pytest.mark.simulation
@pytest.mark.parametrize(
    ["fxp_params", "x1", "x2"], integer_test_data + fractions_test_data
)
def test_mac_hw_for_integers(tmp_path, fxp_params, x1, x2):
    root_dir_path = str(tmp_path)
    root = OnDiskPath("main", parent=root_dir_path)
    mac = MacLayer(fxp_params=fxp_params, vector_width=2)
    test_bench_name = "testbench_fxp_mac"
    y = mac(torch.tensor(x1), torch.tensor(x2)).item()
    testbench = mac.create_testbench(test_bench_name)
    testbench.set_inputs(x1=x1, x2=x2)
    testbench.save_to(root)
    runner = GHDLSimulation(workdir=f"{root_dir_path}", top_design=testbench)
    runner.initialize()
    runner.run()
    actual = testbench.parse_reported_content(runner.getReportedContent())
    assert y == actual


@pytest.mark.skip
@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        # 00001 * 00010 + 00100 * 00001 -> 00000 00010 + 00000 00100 -> 00000 00110 -> 000(00 001)10 -> 00001
        ((0.25, 1.0), (0.5, 0.25), 0.5),
        # 11111 * 00010 + 11100 * 00010 -> 11111 11111 * 00000 00010 + 11111 11100 * 00010
        #  ->  11111 11110
        #    + 11111 11000
        #   -> 11111 10110 -> 111(00 101)10 -> rounding up: 111(00 110)10 ->
        ((-0.25, -1.0), (0.5, 0.5), -0.75),
    ],
)
def test_sw_mac_rounds_towards_zero(x1, x2, expected):
    fxp_params = FXPParams(total_bits=5, frac_bits=2)
    x1 = (0.25, 1.0)
    x2 = (0.5, 0.25)
    mac = MacLayer(fxp_params=fxp_params)
    y = mac(torch.tensor(x1), torch.tensor(x2)).item()
    assert 0.5 == y
