import numpy as np
from torch import Tensor
from torch import nn as nn_torch

from elasticai.creator.nn import fixed_point as nn_creator


def hardtanh_compared_torch(
    total_bits: int, frac_bits: int, min_val=-1.0, max_val=+1.0
) -> None:
    range = [
        -(2 ** (total_bits - frac_bits - 1)),
        2 ** (total_bits - frac_bits - 1) - 1 / 2**frac_bits,
    ]
    stimulus = np.linspace(
        start=range[0], stop=range[1], num=2 ** (total_bits + 1), endpoint=True
    )

    act0 = nn_torch.Hardtanh(min_val=min_val, max_val=max_val)
    out0 = act0(Tensor(stimulus))
    act1 = nn_creator.HardTanh(total_bits, frac_bits, min_val=min_val, max_val=max_val)
    out1 = act1(Tensor(stimulus))

    assert abs(float(sum(out1 - out0))) < 1e-6


def test_hardtanh_compared_torch_6bit_4bit_pm1() -> None:
    hardtanh_compared_torch(total_bits=6, frac_bits=4)


def test_hardtanh_compared_torch_8bit_2bit_pm1() -> None:
    hardtanh_compared_torch(total_bits=8, frac_bits=2)


def test_hardtanh_compared_torch_7bit_4bit_pm1() -> None:
    hardtanh_compared_torch(total_bits=7, frac_bits=4)


def test_hardtanh_compared_torch_7bit_4bit_pm2() -> None:
    hardtanh_compared_torch(total_bits=7, frac_bits=4, min_val=-2.0, max_val=+2.0)
