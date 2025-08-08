import numpy as np
import pytest
from torch import Tensor
from torch import nn as nn_torch

from elasticai.creator.nn import fixed_point as nn_creator


@pytest.mark.parametrize(
    "total_bits, frac_bits",
    [(4, 3), (6, 4), (8, 5), (10, 6), (12, 7), (12, 8), (12, 10)],
)
def test_hard_sigmoid_compared_torch(total_bits: int, frac_bits: int) -> None:
    range = [
        -(2 ** (total_bits - frac_bits - 1)),
        2 ** (total_bits - frac_bits - 1) - 1 / 2**frac_bits,
    ]
    stimulus = np.linspace(
        start=range[0], stop=range[1], num=2 ** (total_bits + 1), endpoint=True
    )

    act0 = nn_torch.Hardsigmoid()
    out0 = act0(Tensor(stimulus))
    act1 = nn_creator.HardSigmoid(total_bits, frac_bits)
    out1 = act1(Tensor(stimulus))
    assert float(sum(abs(out1 - out0))) < 1e-6
