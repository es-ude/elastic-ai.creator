import pytest
import torch

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.vhdl.ghdl_simulation import GHDLSimulation

from .layer import MacLayer

sw_function_test_data = [
    ((1, 1, 1), (-1, 1, -1), -1),
    ((1, -1, -1, 1, 1), (1, -1, 1, -1, 1), 1),
]


@pytest.mark.parametrize(["x1", "x2", "expected"], sw_function_test_data)
def test_sw_function(x1, x2, expected):
    x1 = torch.tensor((1, 1, 1), dtype=torch.float32)
    x2 = torch.tensor((-1, 1, -1), dtype=torch.float32)
    mac = MacLayer(vector_width=x1.shape[0])
    expected = -1
    y = mac(x1, x2).item()
    assert expected == y


test_data = [
    # 00010 * 00010 -> 00000 00100 -> 000(00 001)00 -> 00001
    ((1, 1, 1), (1, 1, 1)),
    # 00010 * 01000 -> 00000 10000 -> 000(00 100)00 -> 00100
    ((-1, 1), (-1, -1)),
    ((-1, -1, 1), (1, 1, 1)),
]


@pytest.mark.simulation
@pytest.mark.parametrize(["x1", "x2"], test_data)
def test_mac_hw_for_integers(tmp_path, x1, x2):
    root_dir_path = str(tmp_path)
    root = OnDiskPath("main", parent=root_dir_path)
    mac = MacLayer(vector_width=len(x1))
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
