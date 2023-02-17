import unittest

from elasticai.creator.nn.hard_sigmoid import HardSigmoid
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl_for_deprecation.translator.abstract import (
    FPHardSigmoidModule,
)
from elasticai.creator.vhdl_for_deprecation.translator.pytorch.build_functions.fp_hard_sigmoid_build_function import (
    build_fp_hard_sigmoid,
)


class FPHardSigmoidBuildFunctionTest(unittest.TestCase):
    def test_build_function_returns_correct_type(self) -> None:
        layer = FixedPointHardSigmoid(
            fixed_point_factory=FixedPoint.get_builder(total_bits=8, frac_bits=4)
        )
        self.assertEqual(type(layer_module), FPHardSigmoidModule)
