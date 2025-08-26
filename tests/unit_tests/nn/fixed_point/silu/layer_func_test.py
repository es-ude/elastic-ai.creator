from os import environ

import pytest
import torch

from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn import fixed_point as nn_creator


@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps",
    [
        (4, 3, 4),
        (6, 4, 32),
        (8, 5, 64),
        (10, 6, 16),
        (12, 7, 128),
        (12, 8, 64),
        (12, 10, 512),
    ],
)
def test_silu_compared_torch(total_bits: int, frac_bits: int, num_steps: int) -> None:
    fxp = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    vrange = (fxp.minimum_as_rational, fxp.maximum_as_rational)
    stimulus = torch.arange(
        start=vrange[0], end=vrange[1], step=fxp.minimum_step_as_rational
    )

    act0 = torch.nn.SiLU()
    out0 = act0(stimulus)
    act1 = nn_creator.SiLU(
        total_bits=total_bits,
        frac_bits=frac_bits,
        num_steps=num_steps,
        sampling_intervall=vrange,
    )
    out1 = act1(stimulus)
    metric_mean_abs = float(sum(abs(out1 - out0))) / stimulus.shape[0]
    metric_mean = float(abs(sum(out1 - out0))) / stimulus.shape[0]

    if environ.get("PLOT_FOR_TESTS", "off") == "on":
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(stimulus, out0, "k", marker=".", label="Torch")
        plt.plot(stimulus, out1, "r", marker=".", label="Creator")
        plt.xlim(vrange)
        plt.legend()
        plt.title(
            f"{total_bits}, {frac_bits}, {num_steps} ({metric_mean_abs:.5f}, {metric_mean: .5f})"
        )
        plt.grid()
        plt.tight_layout()
        plt.show()

    assert metric_mean_abs < 1.2 * fxp.minimum_step_as_rational * (
        out1.max() - out1.min()
    )
    assert metric_mean < 0.6 * fxp.minimum_step_as_rational
