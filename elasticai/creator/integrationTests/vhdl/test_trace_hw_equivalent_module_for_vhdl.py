import unittest
from elasticai.creator.vhdl.model_tracing import Tracer
from elasticai.creator.vhdl.number_representations import ClippedFixedPoint
from elasticai.creator.vhdl.modules import FixedPointLinear, Module


class TestForTracingHWEquivalentModelsToGenerateVHDL(unittest.TestCase):
    def test_treat_single_hw_equivalent_module_as_leaf(self):
        class MyModel(Module):
            def __init__(self):
                super().__init__()
                fp_factory = ClippedFixedPoint.get_factory(total_bits=16, frac_bits=8)
                self.fp_linear = FixedPointLinear(
                    in_features=1, out_features=1, fixed_point_factory=fp_factory
                )

            def forward(self, x):
                return self.fp_linear(x)

        model = MyModel()

        tracer = Tracer()
        self.assertTrue(tracer.is_leaf_module(model.fp_linear, "fp_linear"))
