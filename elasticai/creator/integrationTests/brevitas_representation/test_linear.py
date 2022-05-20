import unittest

import brevitas.nn as bnn
import torch
from torch import nn

import elasticai.creator.brevitas.brevitas_quantizers as bquant
from elasticai.creator.brevitas.translation_functions.linear import (
    translate_linear_layer,
)
from elasticai.creator.layers import Binarize, QLinear, Ternarize


class LinearLayerTest(unittest.TestCase):
    @staticmethod
    def create_qtorch_linear_layer(quantizer):
        return QLinear(in_features=5, out_features=10, bias=True, quantizer=quantizer)

    @staticmethod
    def create_brevitas_linear_layer(weight_quant, bias_quant):
        return bnn.QuantLinear(
            in_features=5,
            out_features=10,
            bias=True,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
        )

    def test_translate_linear_layer_binary_weight_bias_quant(self):
        layer = self.create_qtorch_linear_layer(quantizer=Binarize())
        target = self.create_brevitas_linear_layer(
            weight_quant=bquant.BinaryWeights, bias_quant=bquant.BinaryBias
        )
        translated = translate_linear_layer(layer)

        self.assertIsInstance(translated, bnn.QuantLinear)
        self.assertEqual(target.in_features, translated.in_features)
        self.assertEqual(target.out_features, translated.out_features)
        self.assertIsNotNone(translated.bias)
        self.assertEqual(str(target.weight_quant), str(translated.weight_quant))
        self.assertEqual(str(target.bias_quant), str(translated.bias_quant))

    def test_translate_linear_layer_ternary_weight_bias_quant(self):
        layer = self.create_qtorch_linear_layer(quantizer=Ternarize())
        target = self.create_brevitas_linear_layer(
            weight_quant=bquant.TernaryWeights, bias_quant=bquant.TernaryBias
        )
        translated = translate_linear_layer(layer)

        self.assertIsInstance(translated, bnn.QuantLinear)
        self.assertEqual(target.in_features, translated.in_features)
        self.assertEqual(target.out_features, translated.out_features)
        self.assertIsNotNone(translated.bias)
        self.assertEqual(str(target.weight_quant), str(translated.weight_quant))
        self.assertEqual(str(target.bias_quant), str(translated.bias_quant))

    def test_translate_linear_layer_on_random_input(self):
        layer = self.create_qtorch_linear_layer(quantizer=Ternarize())

        qtorch_model = nn.Sequential(layer)

        translated_linear = translate_linear_layer(layer)
        translated_linear.weight = nn.Parameter(layer.weight)
        translated_linear.bias = nn.Parameter(layer.bias)

        translated_model = nn.Sequential(translated_linear)

        random_input = torch.randn((1, 5))
        qtorch_output = qtorch_model(random_input)
        translated_output = translated_model(random_input)

        self.assertListEqual(qtorch_output.tolist(), translated_output.tolist())


if __name__ == "__main__":
    unittest.main()
