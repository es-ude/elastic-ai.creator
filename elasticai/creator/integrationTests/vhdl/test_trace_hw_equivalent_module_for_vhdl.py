import unittest

from torch import Graph

from elasticai.creator.vhdl.model_tracing import Tracer
from elasticai.creator.vhdl.number_representations import ClippedFixedPoint
from elasticai.creator.vhdl.modules import FixedPointLinear, Module


class FPLinearModel(Module):
    def __init__(self):
        super().__init__()
        fp_factory = ClippedFixedPoint.get_factory(total_bits=16, frac_bits=8)
        self.fp_linear = FixedPointLinear(
            in_features=1, out_features=1, fixed_point_factory=fp_factory
        )

    def forward(self, x):
        return self.fp_linear(x)


class TestForTracingHWEquivalentModelsToGenerateVHDL(unittest.TestCase):
    def test_treat_single_hw_equivalent_module_as_leaf(self):
        model = FPLinearModel()
        tracer = Tracer()
        self.assertTrue(tracer.is_leaf_module(model.fp_linear, "fp_linear"))

    def test_generated_graph_nodes_provide_corresponding_modules(self):
        model = FPLinearModel()
        tracer = Tracer()
        graph: Graph = tracer.trace(model)
        for node in graph.nodes:
            if hasattr(node, "module"):
                module = node.module
        self.assertTrue(module is model.fp_linear)

    def test_graph_for_single_fp_linear_layer_model_contains_no_call_function_nodes(
        self,
    ):
        model = FPLinearModel()
        tracer = Tracer()
        graph = tracer.trace(model)
        self.assertFalse(any(n.op == "call_function" for n in graph.nodes))
