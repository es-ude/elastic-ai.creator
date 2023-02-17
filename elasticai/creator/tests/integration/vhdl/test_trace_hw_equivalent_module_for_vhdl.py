import typing
import unittest

from elasticai.creator.mlframework import Module
from elasticai.creator.vhdl.number_representations import ClippedFixedPoint
from elasticai.creator.vhdl.tracing.hw_equivalent_fx_tracer import HWEquivalentFXTracer
from elasticai.creator.vhdl.translatable_modules.layers import (
    FixedPointLinear,
    RootModule,
)


class FPLinearModel(RootModule):
    def __init__(self):
        super().__init__()
        data_width = 16
        fp_factory = ClippedFixedPoint.get_builder(total_bits=data_width, frac_bits=8)
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
        tracer = HWEquivalentFXTracer()
        graph = tracer.trace(model)
        layer = None
        for node in graph.module_nodes:
            if graph.node_has_module(node):
                layer = graph.get_module_for_node(node)
        self.assertTrue(
            layer is model.fp_linear,
            "expected: {}, actual: {}".format(model.fp_linear, layer),
        )

    def test_graph_for_single_fp_linear_layer_model_contains_no_call_function_nodes(
        self,
    ):
        model = FPLinearModel()
        tracer = HWEquivalentFXTracer()
        graph = tracer.trace(typing.cast(Module, model))
        self.assertFalse(
            any(n.op == "call_function" for n in graph.nodes),
            "; ".join((f"({n.name}, {n.op})" for n in graph.nodes)),
        )

    def test_graph_provides_exactly_one_module_node(self):
        model = FPLinearModelWithGetItemFunctionNode()
        graph = HWEquivalentFXTracer().trace(typing.cast(Module, model))
        nodes = tuple(graph.module_nodes)
        self.assertEqual(model.fp_linear, graph.get_module_for_node(nodes[0].name))
        self.assertEqual(1, len(nodes))
