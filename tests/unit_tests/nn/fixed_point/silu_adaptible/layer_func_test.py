import pytest
import torch
from torch import nn as nn_torch

from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn import fixed_point as nn_creator


@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps",
    [(4, 3, 4), (6, 4, 32), (8, 5, 64), (10, 6, 16), (12, 7, 128), (12, 9, 64)],
)
def test_silu_compared_torch(total_bits: int, frac_bits: int, num_steps: int) -> None:
    fxp = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    vrange = (fxp.minimum_as_rational, fxp.maximum_as_rational)
    stimulus = torch.arange(
        start=vrange[0], end=vrange[1], step=fxp.minimum_step_as_rational
    )

    act0 = nn_torch.SiLU()
    out0 = act0(stimulus).detach().numpy()
    act1 = nn_creator.AdaptableSiLU(
        total_bits=total_bits,
        frac_bits=frac_bits,
        num_steps=num_steps,
        sampling_intervall=vrange,
    )
    out1 = act1(stimulus).detach().numpy()
    metric_mean_abs = float(sum(abs(out1 - out0))) / stimulus.shape[0]
    metric_mean = float(abs(sum(out1 - out0))) / stimulus.shape[0]

    assert metric_mean_abs < 2.2 * fxp.minimum_step_as_rational * (
        out0.max() - out0.min()
    )
    assert metric_mean < 0.6 * fxp.minimum_step_as_rational
