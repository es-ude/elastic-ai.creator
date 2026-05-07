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


@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps, q_input, q_output",
    [
        (3, 2, 8, [-4, -3, -2, -1, 0, 1, 2, 3], [-1, -1, -1, -1, 0, 0, 1, 2]),
        (
            5,
            3,
            16,
            [-16, -14, -12, -10, -8, -6, -4, -2, 1, 3, 5, 7, 9, 11, 13, 15],
            [-2, -2, -2, -2, -2, -2, -2, -1, 0, 1, 2, 4, 6, 8, 10, 12],
        ),
    ],
)
def test_transfer_function_precomputed_adapt_silu(
    total_bits: int,
    frac_bits: int,
    num_steps: int,
    q_input: list[int],
    q_output: list[int],
) -> None:
    act = nn_creator.AdaptableSiLU(
        total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps
    )
    qin_result, qout_result = act.get_lut_integer()
    assert len(qin_result) == num_steps
    assert len(qout_result) == num_steps
    assert q_input == qin_result
    assert q_output == qout_result
