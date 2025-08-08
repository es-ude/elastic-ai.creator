from os import environ

import numpy as np
import pytest
from torch import Tensor
from torch import nn as nn_torch

from elasticai.creator.nn import fixed_point as nn_creator


@pytest.mark.parametrize(
    "total_bits, frac_bits", [(4, 3), (6, 4), (8, 5), (10, 6), (12, 7), (12, 8)]
)
def test_sigmoid_compared_torch(total_bits: int, frac_bits: int) -> None:
    plot_results = environ.get(["PLOT_TESTING"], "off") == "on"
    num_steps = 2 ** (total_bits + 1)
    range = (
        -(2 ** (total_bits - frac_bits - 1)),
        2 ** (total_bits - frac_bits - 1) - 1 / 2**frac_bits,
    )
    stimulus = np.linspace(start=range[0], stop=range[1], num=num_steps, endpoint=True)

    act0 = nn_torch.Sigmoid()
    out0 = act0(Tensor(stimulus))
    act1 = nn_creator.Sigmoid(total_bits, frac_bits, num_steps, range)
    out1 = act1(Tensor(stimulus))

    if plot_results:
        from matplotlib import pyplot as plt

        plt.title(f"Sigmoid (n={total_bits}, {frac_bits})")
        plt.plot(stimulus, out0, color="k", label="Torch", marker=".", markersize=4)
        plt.step(
            stimulus,
            out1,
            color="r",
            label="Creator",
            where="mid",
            marker=".",
            markersize=4,
        )

        plt.ylabel("Output")
        plt.xlabel("Input")
        plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.ylim([-0.02, +1.02])
        plt.xlim(range)
        plt.xticks([range[0], range[0] / 2, 0.0, range[1] / 2, range[1]])
        plt.grid()
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
    assert float(sum(abs(out1 - out0))) < 1e-6
