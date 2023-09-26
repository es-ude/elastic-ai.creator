import dataclasses

import pytest
import torch

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.vhdl.ghdl_simulation import GHDLSimulation

from .sw_function import MacLayer
from .testbench import TestBench


@dataclasses.dataclass
class FXPParams:
    total_bits: int
    frac_bits: int


integer_test_data = [
    (FXPParams(4, 0), x1, x2)
    for x1, x2 in [
        ((1.0, 0.0), (-1.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0)),
        ((1.0, 0.0), (3.0, 0.0)),
        ((2.0, 0.0), (4.0, 0.0)),
        ((-2.0, 0.0), (4.0, 0.0)),
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
    ]
]


@pytest.mark.simulation
@pytest.mark.parametrize(
    ["fxp_params", "x1", "x2"], integer_test_data + fractions_test_data
)
def test_mac_hw_for_integers(tmp_path, fxp_params, x1, x2):
    root_dir_path = str(tmp_path)
    root = OnDiskPath("main", parent=root_dir_path)
    mac = MacLayer(total_bits=fxp_params.total_bits, frac_bits=fxp_params.frac_bits)
    test_bench_name = "testbench_fxp_mac"
    y = mac(torch.tensor(x1), torch.tensor(x2)).item()

    testbench = TestBench(
        total_bits=fxp_params.total_bits,
        frac_bits=fxp_params.frac_bits,
        x1=x1,
        x2=x2,
        name=test_bench_name,
        vector_width=len(x1),
    )
    mac_design = mac.create_design()
    testbench.save_to(root)
    mac_design.save_to(root)
    runner = GHDLSimulation(workdir=f"{root_dir_path}", top_design=testbench)
    runner.initialize()
    runner.run()
    actual = testbench.parse_reported_content(runner.getReportedContent())
    assert y == actual


@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        # 00001 * 00010 + 00100 * 00001 -> 00000 00010 + 00000 00100 -> 00000 00110 -> 000(00 001)10 -> 00(000 10)00 -> 00010
        ((0.25, 1.0), (0.5, 0.25), 0.5),
        # 00001 * 00010 + 00100 * 00010 -> 00000 00010 + 00000 01000 -> 00000 01010 -> 000(00 010)10 -> 00010
        ((0.25, 1.0), (0.5, 0.5), 0.5),
    ],
)
def test_sw_mac_rounds_half_to_even(x1, x2, expected):
    fxp_params = FXPParams(total_bits=5, frac_bits=2)
    x1 = (0.25, 1.0)
    x2 = (0.5, 0.25)
    mac = MacLayer(total_bits=fxp_params.total_bits, frac_bits=fxp_params.frac_bits)
    y = mac(torch.tensor(x1), torch.tensor(x2)).item()
    assert 0.5 == y
