import unittest
import typing
from elasticai.creator.vhdl.hw_equivalent_layers import FixedPointLinear
from elasticai.creator.vhdl.model_tracing import HWEquivalentTracer
from elasticai.creator.vhdl.number_representations import ClippedFixedPoint
from mlframework import Module
from vhdl.hw_equivalent_layers import RootModule


class FPLinearModel(RootModule):
    def __init__(self):
        super().__init__()
        data_width = 16
        fp_factory = ClippedFixedPoint.get_factory(total_bits=data_width, frac_bits=8)
        self.fp_linear = FixedPointLinear(
            in_features=1,
            out_features=1,
            fixed_point_factory=fp_factory,
            data_width=data_width,
        )

    def forward(self, x):
        return self.fp_linear(x)


class FPLinearModelWithGetItemFunctionNode(FPLinearModel):
    def forward(self, x):
        return self.fp_linear(x[:2])


class TestForTracingHWEquivalentModelsToGenerateVHDL(unittest.TestCase):
    def test_generated_graph_nodes_provide_corresponding_modules(self):
        model = FPLinearModel()
        tracer = HWEquivalentTracer()
        graph = tracer.trace(typing.cast(Module, model))
        module = None
        for node in graph.nodes:
            if hasattr(node, "module"):
                print(node)
                module = node.module
        self.assertTrue(module is model.fp_linear)

    def test_graph_for_single_fp_linear_layer_model_contains_no_call_function_nodes(
        self,
    ):
        model = FPLinearModel()
        tracer = HWEquivalentTracer()
        graph = tracer.trace(model)
        self.assertFalse(
            any(n.op == "call_function" for n in graph.nodes),
            "; ".join((f"({n.name}, {n.op})" for n in graph.nodes)),
        )

    def test_graph_provides_exactly_one_module_node(self):
        model = FPLinearModelWithGetItemFunctionNode()
        graph = HWEquivalentTracer().trace(typing.cast(Module, model))
        nodes = tuple(graph.module_nodes)
        self.assertEqual(model.fp_linear, nodes[0].module)
        self.assertEqual(1, len(nodes))
