import numpy as np
from torch import Tensor
from torch import nn as nn_torch

from elasticai.creator.nn import fixed_point as nn_creator


def hard_sigmoid_compared_torch(total_bits: int, frac_bits: int) -> None:
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

    assert abs(float(sum(out1 - out0))) < 1e-6


def test_hard_sigmoid_4bit_2bit() -> None:
    hard_sigmoid_compared_torch(total_bits=4, frac_bits=2)


def test_hard_sigmoid_4bit_3bit() -> None:
    hard_sigmoid_compared_torch(total_bits=4, frac_bits=3)


def test_hard_sigmoid_6bit_4bit() -> None:
    hard_sigmoid_compared_torch(total_bits=6, frac_bits=4)


def test_hard_sigmoid_8bit_4bit() -> None:
    hard_sigmoid_compared_torch(total_bits=8, frac_bits=4)


def test_hard_sigmoid_8bit_6bit() -> None:
    hard_sigmoid_compared_torch(total_bits=8, frac_bits=6)


def test_hard_sigmoid_12bit_6bit() -> None:
    hard_sigmoid_compared_torch(total_bits=12, frac_bits=6)


def test_hard_sigmoid_12bit_8bit() -> None:
    hard_sigmoid_compared_torch(total_bits=12, frac_bits=8)


def test_hard_sigmoid_12bit_10bit() -> None:
    hard_sigmoid_compared_torch(total_bits=12, frac_bits=10)
