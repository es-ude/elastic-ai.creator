import pytest
import torch

from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn import fixed_point as nn_creator


@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps",
    [(4, 3, 4), (6, 4, 32), (8, 5, 64), (10, 6, 16), (12, 7, 128), (12, 11, 64)],
)
def test_relu_compared_torch(total_bits: int, frac_bits: int, num_steps: int) -> None:
    fxp = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    vrange = (fxp.minimum_as_rational, fxp.maximum_as_rational)
    stimulus = torch.arange(
        start=vrange[0], end=vrange[1], step=fxp.minimum_step_as_rational
    )

    act0 = torch.nn.ReLU()
    out0 = act0(stimulus)
    act1 = nn_creator.ReLU(total_bits)
    out1 = act1(stimulus)
    act2 = nn_creator.ReLU(total_bits, True)
    out2 = act2(stimulus)

    assert float(sum(abs(out1 - out0))) < 1e-6
    assert float(sum(abs(out2 - out0))) < 1e-6
