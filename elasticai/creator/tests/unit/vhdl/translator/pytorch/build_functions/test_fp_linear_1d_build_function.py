import unittest

import torch

from elasticai.creator.nn.linear import FixedPointLinear
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.pytorch.build_functions import build_fp_linear_1d


def aragnge_parameter(
    start: int, end: int, shape: tuple[int, ...]
) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.reshape(torch.arange(start, end, dtype=torch.float32), shape)
    )


class FPLinear1dBuildFunctionTest(unittest.TestCase):
    def test_weights_and_bias_correct_set(self) -> None:
        fp_factory = FixedPoint.get_factory(total_bits=8, frac_bits=4)

        linear = FixedPointLinear(
            fixed_point_factory=fp_factory, in_features=3, out_features=2, bias=True
        )
        linear.weight = aragnge_parameter(start=1, end=4, shape=(1, -1))
        linear.bias = aragnge_parameter(start=1, end=2, shape=(-1,))

        fp_linear1d = build_fp_linear_1d(
            linear, layer_id="ll1", work_library_name="work"
        )
        self.assertEqual(fp_linear1d.weight, [[1.0, 2.0, 3.0]])
        self.assertEqual(fp_linear1d.bias, [1.0])
