import pytest
import torch

from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn.fixed_point.mac.layer import MacLayer


@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        # 00001 * 00010 + 00100 * 00001 -> 00000 00010 + 00000 00100 -> 00000 00110
        ((0.25, 1.0), (0.5, 0.25), 0.375),
        # 11111 * 00010 + 11100 * 00010 -> 11111 11111 * 00000 00010 + 11111 11100 * 00000 00010
        #  ->  11111 11110
        #    + 11111 11000
        #   -> 11111 10110 -> 11111 10110
        ((-0.25, -1.0), (0.5, 0.5), -0.625),
    ],
)
def test_sw_mac_rounds_towards_zero(x1, x2, expected):
    fxp_params = FxpParams(total_bits=5, frac_bits=2)
    mac = MacLayer(fxp_params=fxp_params, vector_width=len(x1))
    y = mac(torch.tensor(x1), torch.tensor(x2)).item()
    assert expected == y
