import pytest
import torch

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.vhdl.ghdl_simulation import GHDLSimulation

from .sw_function import MacLayer
from .testbench import TestBench


@pytest.mark.simulation
def test_mac_hw():
    root_name = "hw_test"
    root = OnDiskPath(root_name)
    x1 = (0.0, 1.0)
    x2 = (0.0, -1.0)
    mac = MacLayer(total_bits=4, frac_bits=2)
    test_bench_name = "testbench_fxp_mac"
    y = mac(torch.tensor(x1), torch.tensor(x2)).item()

    testbench = TestBench(
        total_bits=4,
        frac_bits=2,
        x1=x1,
        x2=x2,
        name=test_bench_name,
    )
    mac_design = mac.create_design()
    testbench.save_to(root)
    mac_design.save_to(root)
    runner = GHDLSimulation(workdir=f"{root_name}", top_design=testbench)
    runner.initialize()
    runner.run()
    actual = testbench.parse_reported_content(runner.getReportedContent())
    assert y == actual
