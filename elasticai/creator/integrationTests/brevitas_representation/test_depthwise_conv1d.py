import unittest

import brevitas.nn as bnn
import torch
from torch import nn

import elasticai.creator.brevitas.brevitas_quantizers as bquant
from elasticai.creator.brevitas.translation_functions.conv import translate_conv1d
from elasticai.creator.integrationTests.brevitas_representation.conv_params_comparison import (
    ConvTest,
)
from elasticai.creator.layers import Binarize, QConv1d, Ternarize


# When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also known as a “depthwise convolution”.
class DepthwiseConv1dTest(ConvTest):
    @staticmethod
    def create_qtorch_conv_layers(
        in_channel, quantizer, padding=0, padding_mode="zeros"
    ):
        return QConv1d(
            in_channels=in_channel,
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
    def create_brevitas_conv_layer(
        in_channel, weight_quant, bias_quant, padding_type="standard"
    ):
        return bnn.QuantConv1d(
            in_channels=in_channel,
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

    def test_depthwise_conv1d_binary_weight_bias_quant(self):
        in_channel = 3
        layer = self.create_qtorch_conv_layers(
            in_channel=in_channel, quantizer=Binarize()
        )
        target = self.create_brevitas_conv_layer(
            in_channel=in_channel,
            weight_quant=bquant.BinaryWeights,
            bias_quant=bquant.BinaryBias,
        )
        translated = translate_conv1d(layer)

        self.assertConv1dParams(target, translated)

    def test_depthwise_conv1d_ternary_weight_bias_quant(self):
        in_channel = 3
        layer = self.create_qtorch_conv_layers(
            in_channel=in_channel, quantizer=Ternarize()
        )
        target = self.create_brevitas_conv_layer(
            in_channel=in_channel,
            weight_quant=bquant.TernaryWeights,
            bias_quant=bquant.TernaryBias,
        )
        translated = translate_conv1d(layer)

        self.assertConv1dParams(target, translated)

    def test_depthwise_conv1d_padding(self):
        in_channel = 3
        layer = self.create_qtorch_conv_layers(
            in_channel=in_channel, quantizer=Binarize(), padding="same"
        )
        target = self.create_brevitas_conv_layer(
            in_channel=in_channel,
            weight_quant=bquant.BinaryWeights,
            bias_quant=bquant.BinaryBias,
            padding_type="same",
        )
        translated = translate_conv1d(layer)

        self.assertConv1dParams(target, translated)

    def test_depthwise_conv1d_wrong_padding_mode(self):
        in_channel = 3
        layer = self.create_qtorch_conv_layers(
            in_channel=in_channel, quantizer=Binarize(), padding_mode="reflect"
        )

        self.assertRaises(NotImplementedError, translate_conv1d, layer)

    def test_depthwise_conv1d_on_random_input(self):
        in_channel = 5
        layer = self.create_qtorch_conv_layers(
            in_channel=in_channel, quantizer=Ternarize()
        )

        qtorch_model = nn.Sequential(layer)

        translated_conv1d = translate_conv1d(layer)
        translated_conv1d.weight = nn.Parameter(layer.weight)
        translated_conv1d.bias = nn.Parameter(layer.bias)

        translated_model = nn.Sequential(translated_conv1d)

        random_input = torch.randn((1, 5, 1000))
        qtorch_output = qtorch_model(random_input)
        translated_output = translated_model(random_input)

        self.assertListEqual(qtorch_output.tolist(), translated_output.tolist())


if __name__ == "__main__":
    unittest.main()
