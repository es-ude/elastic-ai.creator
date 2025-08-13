import pytest
import torch

from elasticai.creator.nn import fixed_point as nn_creator
from elasticai.creator.nn.fixed_point.math_operations import FixedPointConfig


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
def test_prelu_compared_torch(total_bits: int, frac_bits: int, num_steps: int) -> None:
    fxp = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
    vrange = (fxp.minimum_as_rational, fxp.maximum_as_rational)
    stimulus = torch.arange(
        start=vrange[0],
        end=vrange[1] + fxp.minimum_step_as_rational,
        step=fxp.minimum_step_as_rational,
    )

    act0 = torch.nn.PReLU()
    out0 = act0(stimulus).detach().numpy()
    act1 = nn_creator.PReLU(
        total_bits=total_bits,
        frac_bits=frac_bits,
        num_steps=num_steps,
        sampling_intervall=vrange,
    )
    out1 = act1(stimulus).detach().numpy()
    metric_mean_abs = float(sum(abs(out1 - out0))) / stimulus.shape[0]
    metric_mean = float(abs(sum(out1 - out0))) / stimulus.shape[0]

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(stimulus, out0, 'k', marker='.', label='Torch')
    # plt.plot(stimulus, out1, 'r', marker='.', label='Creator')
    # plt.xlim(vrange)
    # plt.legend()
    # plt.title(f"{total_bits}, {frac_bits}, {num_steps} ({metric_mean_abs:.5f}, {metric_mean: .5f})")
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    print(metric_mean_abs, metric_mean)
    assert metric_mean_abs < 1.5 * fxp.minimum_step_as_rational * (
        out1.max() - out1.min()
    )
    assert metric_mean < 0.7 * fxp.minimum_step_as_rational
