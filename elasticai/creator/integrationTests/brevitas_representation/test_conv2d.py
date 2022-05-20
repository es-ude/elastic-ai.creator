import unittest

import brevitas.nn as bnn
import torch
from torch import nn

import elasticai.creator.brevitas.brevitas_quantizers as bquant
from elasticai.creator.brevitas.translation_functions.conv import translate_conv2d
from elasticai.creator.integrationTests.brevitas_representation.conv_params_comparison import (
    ConvTest,
)
from elasticai.creator.layers import Binarize, QConv2d, Ternarize


class Conv2dTest(ConvTest):
    @staticmethod
    def create_qtorch_conv_layers(quantizer, padding=0, padding_mode="zeros"):
        return QConv2d(
            in_channels=5,
            out_channels=10,
            kernel_size=10,
            stride=1,
            padding=padding,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode=padding_mode,
            quantizer=quantizer,
        )

    @staticmethod
    def create_brevitas_conv_layer(weight_quant, bias_quant, padding_type="standard"):
        return bnn.QuantConv2d(
            in_channels=5,
            out_channels=10,
            kernel_size=10,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_type=padding_type,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
        )

    def test_conv2d_binary_weight_bias_quant(self):
        layer = self.create_qtorch_conv_layers(quantizer=Binarize())
        target = self.create_brevitas_conv_layer(
            weight_quant=bquant.BinaryWeights, bias_quant=bquant.BinaryBias
        )
        translated = translate_conv2d(layer)

        self.assertConv2dParams(target, translated)

    def test_conv2d_ternaryweight_bias_quant(self):
        layer = self.create_qtorch_conv_layers(quantizer=Ternarize())
        target = self.create_brevitas_conv_layer(
            weight_quant=bquant.TernaryWeights, bias_quant=bquant.TernaryBias
        )
        translated = translate_conv2d(layer)

        self.assertConv2dParams(target, translated)

    def test_conv2d_padding(self):
        layer = self.create_qtorch_conv_layers(quantizer=Binarize(), padding="same")
        target = self.create_brevitas_conv_layer(
            weight_quant=bquant.BinaryWeights,
            bias_quant=bquant.BinaryBias,
            padding_type="same",
        )
        translated = translate_conv2d(layer)

        self.assertConv2dParams(target, translated)

    def test_conv2d_wrong_padding_mode(self):
        layer = self.create_qtorch_conv_layers(
            quantizer=Binarize(), padding_mode="reflect"
        )
        self.assertRaises(NotImplementedError, translate_conv2d, layer)

    def test_conv2d_on_random_input(self):
        layer = self.create_qtorch_conv_layers(quantizer=Ternarize())
        qtorch_model = nn.Sequential(layer)

        translated_conv2d = translate_conv2d(layer)
        translated_conv2d.weight = nn.Parameter(layer.weight)
        translated_conv2d.bias = nn.Parameter(layer.bias)

        translated_model = nn.Sequential(translated_conv2d)

        random_input = torch.randn((10, 5, 10, 10))
        qtorch_output = qtorch_model(random_input)
        translated_output = translated_model(random_input)

        self.assertListEqual(qtorch_output.tolist(), translated_output.tolist())


if __name__ == "__main__":
    unittest.main()
