import unittest

from elasticai.creator.nn.relu import ReLU
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers.fp_relu_module import (
    FPReLUModule,
)
from elasticai.creator.vhdl.translator.pytorch.build_functions.fp_relu_build_function import (
    build_fp_relu,
)


class FPReluBuildFunctionTest(unittest.TestCase):
    def test_build_function_returns_correct_type(self) -> None:
        fp_factory = FixedPoint.get_factory(total_bits=8, frac_bits=4)
        layer = ReLU()
        layer_module = build_fp_relu(
            layer, layer_id="relu1", fixed_point_factory=fp_factory
        )
        self.assertEqual(type(layer_module), FPReLUModule)
