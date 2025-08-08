import numpy as np
from torch import Tensor
from torch import nn as nn_torch

from elasticai.creator.nn import fixed_point as nn_creator


def relu_compared_torch(total_bits: int, frac_bits: int) -> None:
    range = [
        -(2 ** (total_bits - frac_bits - 1)),
        2 ** (total_bits - frac_bits - 1) - 1 / 2**frac_bits,
    ]
    stimulus = np.linspace(
        start=range[0], stop=range[1], num=2 ** (total_bits + 1), endpoint=True
    )

    act0 = nn_torch.ReLU()
    out0 = act0(Tensor(stimulus))
    act1 = nn_creator.ReLU(total_bits)
    out1 = act1(Tensor(stimulus))
    act2 = nn_creator.ReLU(total_bits, True)
    out2 = act2(Tensor(stimulus))

    assert float(sum(abs(out1 - out0))) < 1e-6
    assert float(sum(abs(out2 - out0))) < 1e-6


def test_relu_comparison_4bit_2bit() -> None:
    relu_compared_torch(total_bits=4, frac_bits=2)


def test_relu_comparison_4bit_4bit() -> None:
    relu_compared_torch(total_bits=4, frac_bits=4)


def test_relu_comparison_6bit_4bit() -> None:
    relu_compared_torch(total_bits=6, frac_bits=4)


def test_relu_comparison_8bit_4bit() -> None:
    relu_compared_torch(total_bits=8, frac_bits=4)
