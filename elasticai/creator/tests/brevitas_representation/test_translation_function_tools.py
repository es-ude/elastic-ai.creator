import unittest
from types import SimpleNamespace

import elasticai.creator.brevitas.brevitas_quantizers as bquant
from elasticai.creator.brevitas.translation_functions.translation_function_tools import (
    set_quantizers,
)
from elasticai.creator.layers import Binarize, Ternarize


class LayerMock:
    def __init__(self, quantizer):
        self.parametrizations = SimpleNamespace(weight=[quantizer])


class ConversionFunctionToolsTest(unittest.TestCase):
    def create_layers(self, quantizer):
        return LayerMock(quantizer=quantizer)

    def create_dict(self, bias):
        args = {"bias": bias}
        old_args = args.copy()
        return args, old_args

    def test_set_quantizers_binarize_quantizer_no_bias(self):
        layer = self.create_layers(quantizer=Binarize())
        args, old_args = self.create_dict(bias=False)
        set_quantizers(layer, args=args)
        self.assertNotEqual(old_args, args)
        old_args["weight_quant"] = bquant.BinaryWeights
        self.assertDictEqual(old_args, args)

    def test_set_quantizers_binarize_quantizer_with_bias(self):
        layer = self.create_layers(quantizer=Binarize())
        args, old_args = self.create_dict(bias=True)
        set_quantizers(layer, args=args)
        self.assertNotEqual(old_args, args)
        old_args["weight_quant"] = bquant.BinaryWeights
        old_args["bias_quant"] = bquant.BinaryBias
        self.assertDictEqual(old_args, args)

    def test_set_quantizers_ternarize_quantizer_no_bias(self):
        layer = self.create_layers(quantizer=Ternarize())
        args, old_args = self.create_dict(bias=False)
        set_quantizers(layer, args=args)
        self.assertNotEqual(old_args, args)
        old_args["weight_quant"] = bquant.TernaryWeights
        self.assertDictEqual(old_args, args)

    def test_set_quantizers_ternarize_quantizer_with_bias(self):
        layer = self.create_layers(quantizer=Ternarize())
        args, old_args = self.create_dict(bias=True)
        set_quantizers(layer, args=args)
        self.assertNotEqual(old_args, args)
        old_args["weight_quant"] = bquant.TernaryWeights
        old_args["bias_quant"] = bquant.TernaryBias
        self.assertDictEqual(old_args, args)


if __name__ == "__main__":
    unittest.main()
