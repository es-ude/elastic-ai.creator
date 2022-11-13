import unittest

import torch

from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.quantized_modules.linear import FixedPointLinear
from elasticai.creator.vhdl.translator.pytorch.build_functions import build_fp_linear_1d


def aragnge_parameter(
    start: int, end: int, shape: tuple[int, ...]
) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.reshape(torch.arange(start, end, dtype=torch.float32), shape)
    )


class FPLinear1dBuildFunctionTest(unittest.TestCase):
    def setUp(self) -> None:
        to_fp = FixedPoint.get_factory(total_bits=8, frac_bits=4)

        self.linear = FixedPointLinear(
            layer_name="ll1", fixed_point_factory=to_fp, in_features=3, out_features=2
        )
        self.linear.weight = aragnge_parameter(start=1, end=4, shape=(1, -1))
        self.linear.bias = aragnge_parameter(start=1, end=2, shape=(-1,))

    def test_weights_and_bias_correct_set(self) -> None:
        fplinear1d = build_fp_linear_1d(self.linear)
        self.assertEqual(fplinear1d.weight, [[1.0, 2.0, 3.0]])
        self.assertEqual(fplinear1d.bias, [1.0])
