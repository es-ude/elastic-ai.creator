import pytest
import torch

from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn import fixed_point as nn_creator


@pytest.mark.parametrize(
    "total_bits, frac_bits, init",
    [
        (3, 2, 0.25),
        (4, 2, 0.25),
        (4, 3, 0.25),
        (6, 4, 0.5),
        (8, 5, 0.125),
        (10, 6, 0.125),
        (12, 7, 0.125),
        (12, 8, 0.0625),
        (12, 10, 0.03125),
    ],
)
def test_prelu2_compared_torch(total_bits: int, frac_bits: int, init: float) -> None:
    fxp = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    vrange = (fxp.minimum_as_rational, fxp.maximum_as_rational)
    stimulus = torch.arange(
        start=vrange[0],
        end=vrange[1] + fxp.minimum_step_as_rational,
        step=fxp.minimum_step_as_rational,
    )

    act0 = torch.nn.PReLU(init=init)
    out0 = act0(stimulus).detach().numpy()
    act1 = nn_creator.PReLU2(
        total_bits=total_bits,
        frac_bits=frac_bits,
        init=init,
    )
    out1 = act1(stimulus).detach().numpy()
    metric_mean_abs = float(sum(abs(out1 - out0))) / stimulus.shape[0]
    metric_mean = float(abs(sum(out1 - out0))) / stimulus.shape[0]

    assert metric_mean_abs < 0.5 * fxp.minimum_step_as_rational * (
        out1.max() - out1.min()
    )
    assert metric_mean < 0.25 * fxp.minimum_step_as_rational
