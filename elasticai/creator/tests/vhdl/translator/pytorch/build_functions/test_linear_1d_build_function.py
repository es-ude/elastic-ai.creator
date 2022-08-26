import unittest

import torch

from elasticai.creator.vhdl.translator.pytorch.build_functions import build_linear_1d


def arange_parameter(
    start: int, end: int, shape: tuple[int, ...]
) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.reshape(torch.arange(start, end, dtype=torch.float32), shape)
    )


class Linear1dBuildFunctionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.linear = torch.nn.Linear(in_features=3, out_features=1)
        self.linear.weight = arange_parameter(start=1, end=4, shape=(1, -1))
        self.linear.bias = arange_parameter(start=1, end=2, shape=(-1,))

    def test_weights_and_bias_correct_set(self) -> None:
        linear1d = build_linear_1d(self.linear)
        self.assertEqual(linear1d.weight, [[1.0, 2.0, 3.0]])
        self.assertEqual(linear1d.bias, [1.0])
