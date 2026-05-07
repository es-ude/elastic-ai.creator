import pytest
import torch

from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn import fixed_point as nn_creator


@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps",
    [
        (3, 2, 8),
        (4, 2, 4),
        (4, 3, 4),
        (6, 4, 32),
        (8, 5, 64),
        (10, 6, 16),
        (12, 7, 128),
        (12, 8, 64),
        (12, 10, 512),
    ],
)
def test_tanh_compared_torch(total_bits: int, frac_bits: int, num_steps: int) -> None:
    fxp = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    vrange = (fxp.minimum_as_rational, fxp.maximum_as_rational)
    stimulus = torch.arange(
        start=vrange[0],
        end=vrange[1] + fxp.minimum_step_as_rational,
        step=fxp.minimum_step_as_rational,
    )

    act0 = torch.nn.Tanh()
    out0 = act0(stimulus)
    act1 = nn_creator.Tanh(
        total_bits=total_bits,
        frac_bits=frac_bits,
        num_steps=num_steps,
        sampling_intervall=vrange,
    )
    out1 = act1(stimulus)
    metric_mean_abs = float(sum(abs(out1 - out0))) / stimulus.shape[0]
    metric_mean = float(abs(sum(out1 - out0))) / stimulus.shape[0]

    print(metric_mean_abs, metric_mean)
    assert metric_mean_abs < 1.5 * fxp.minimum_step_as_rational * (
        out1.max() - out1.min()
    )
    assert metric_mean < 0.6 * fxp.minimum_step_as_rational


@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps, q_input, q_output",
    [
        (
            4,
            2,
            16,
            [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
            [-4, -4, -4, -4, -3, -3, -2, -1, 0, 0, 1, 2, 3, 3, 4, 4],
        ),
        (
            6,
            3,
            16,
            [-32, -28, -24, -19, -15, -11, -7, -3, 2, 6, 10, 14, 18, 23, 27, 31],
            [-8, -8, -8, -8, -8, -7, -7, -5, 0, 3, 6, 7, 8, 8, 8, 8],
        ),
    ],
)
def test_transfer_function_precomputed_tanh(
    total_bits: int,
    frac_bits: int,
    num_steps: int,
    q_input: list[int],
    q_output: list[int],
) -> None:
    act = nn_creator.Tanh(
        total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps
    )
    qin_result, qout_result = act.get_lut_integer()
    assert len(qin_result) == num_steps
    assert len(qout_result) == num_steps
    assert q_input == qin_result
    assert q_output == qout_result
