import pytest
import torch

from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn.fixed_point.mac.layer import MacLayer
from elasticai.creator.testing import GHDLSimulator

integer_test_data = [
    (FxpParams(4, 0), x1, x2)
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
    (FxpParams(5, 2), x1, x2)
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


@pytest.mark.skip(
    reason="Not changed for new SimulatedLayer. Needs to be implemented in the future"
)
@pytest.mark.simulation
@pytest.mark.parametrize(
    ["fxp_params", "x1", "x2"], integer_test_data + fractions_test_data
)
def test_mac_hw_for_integers(tmp_path, fxp_params, x1, x2):
    root_dir_path = str(tmp_path)
    mac = MacLayer(fxp_params=fxp_params, vector_width=2)
    y = mac(torch.tensor(x1), torch.tensor(x2)).item()
    sim = mac.create_simulation(GHDLSimulator, root_dir_path)
    actual = sim(x1, x2)
    assert y == actual


@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        # 00001 * 00010 + 00100 * 00001 -> 00000 00010 + 00000 00100 -> 00000 00110 -> 000(00 001)10 -> 00001
        ((0.25, 1.0), (0.5, 0.25), 0.25),
        # 11111 * 00010 + 11100 * 00010 -> 11111 11111 * 00000 00010 + 11111 11100 * 00000 00010
        #  ->  11111 11110
        #    + 11111 11000
        #   -> 11111 10110 -> 111(11 101)10 -> rounding up: 111(11 110)10 -> 11110
        ((-0.25, -1.0), (0.5, 0.5), -0.5),
    ],
)
def test_sw_mac_rounds_towards_zero(x1, x2, expected):
    fxp_params = FxpParams(total_bits=5, frac_bits=2)
    mac = MacLayer(fxp_params=fxp_params, vector_width=len(x1))
    y = mac(torch.tensor(x1), torch.tensor(x2)).item()
    assert expected == y
