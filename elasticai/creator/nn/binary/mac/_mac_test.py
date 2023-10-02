import pytest
import torch
from creator.vhdl.simulated_layer import SimulatedLayer

from elasticai.creator.vhdl.ghdl_simulation import GHDLSimulator

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
    mac = MacLayer(vector_width=len(x1))
    sim = mac.create_simulation(GHDLSimulator, working_dir=root_dir_path)
    actual = sim(x1, x2)
    y = mac(
        torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)
    )
    assert y == actual
