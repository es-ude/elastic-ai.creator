import unittest

from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.quantized_modules.relu import FixedPointReLU
from elasticai.creator.vhdl.translator.abstract.layers.fp_relu_module import (
    FPReLUModule,
)
from elasticai.creator.vhdl.translator.pytorch.build_functions.fp_relu_build_function import (
    build_fp_relu,
)


class FPReluBuildFunctionTest(unittest.TestCase):
    def test_build_function_returns_correct_type(self) -> None:
        layer = FixedPointReLU(
            fixed_point_factory=FixedPoint.get_factory(total_bits=8, frac_bits=4)
        )
        layer_module = build_fp_relu(layer)
        self.assertEqual(type(layer_module), FPReLUModule)
