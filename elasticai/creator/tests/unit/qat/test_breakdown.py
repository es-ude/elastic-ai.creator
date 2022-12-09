import unittest

import torch
from torch.nn.utils import parametrize

from elasticai.creator.precomputation.breakdown import (
    BreakdownConv2dBlock,
    depthwisePointwiseBreakdownConv2dBlock,
    generate_conv2d_sequence_with_width,
)
from elasticai.creator.qat.blocks import Conv2dBlock
from elasticai.creator.qat.layers import Binarize, ChannelShuffle
from elasticai.creator.qat.masks import FixedOffsetMask4d


class BreakdownTest(unittest.TestCase):
    def compare_models_and_weight_shape(self, expected, actual):
        self.assertTrue(len(expected) == len(actual), "number of layers")
        for layer_expected, layer_actual in zip(expected, actual):
            if not hasattr(
                layer_expected, "parametrizations"
            ):  # parametrizations seems to create a dynamic class, impossible to compare instances
                self.assertTrue(
                    isinstance(layer_expected, type(layer_expected)),
                    f"layer types differ expected:{layer_expected} actual:{layer_actual}",
                )
            if hasattr(layer_actual, "conv2d"):
                self.assertSequenceEqual(
                    layer_actual.conv2d.weight.shape,
                    layer_expected.conv2d.weight.shape,
                    f"layer shapes differ expected:{layer_expected.conv2d.weight.shape} actual:{layer_actual.conv2d.weight.shape}",
                )
            if hasattr(layer_actual, "groups"):
                self.assertEqual(
                    layer_actual.groups,
                    layer_expected.groups,
                    f"groups differ expected:{layer_expected.groups} actual:{layer_actual.groups}",
                )

    def test_generate_conv2d_sequence_with_width_base(self):
        layers = generate_conv2d_sequence_with_width(
            in_channels=2,
            out_channels=4,
            activation=torch.nn.Identity(),
            weight_quantization=Binarize(),
            kernel_size=3,
            channel_width=2,
        )
        expected = torch.nn.Sequential(
            Conv2dBlock(
                in_channels=2,
                out_channels=4,
                activation=torch.nn.Identity(),
                conv_quantizer=Binarize(),
                kernel_size=3,
            ),
            ChannelShuffle(groups=1),
        )
        self.compare_models_and_weight_shape(layers, expected)

    def test_generate_conv2d_sequence_with_width_more_complex(self):
        layers = generate_conv2d_sequence_with_width(
            in_channels=4,
            out_channels=2,
            activation=torch.nn.Identity(),
            weight_quantization=Binarize(),
            kernel_size=3,
            channel_width=2,
        )
        expected = torch.nn.Sequential(
            Conv2dBlock(
                in_channels=4,
                out_channels=4,
                activation=torch.nn.Identity(),
                conv_quantizer=Binarize(),
                kernel_size=3,
                groups=2,
            ),
            ChannelShuffle(groups=2),
            Conv2dBlock(
                in_channels=4,
                out_channels=2,
                activation=torch.nn.Identity(),
                conv_quantizer=Binarize(),
                kernel_size=3,
                groups=2,
            ),
        )
        self.compare_models_and_weight_shape(layers, expected)

    def test_generate_conv2d_sequence_with_width_more_than_2_last_groups_unequal_out_channels(
        self,
    ):
        layers = generate_conv2d_sequence_with_width(
            in_channels=256,
            out_channels=256,
            activation=torch.nn.Identity(),
            weight_quantization=Binarize(),
            kernel_size=3,
            channel_width=8,
        )
        expected = torch.nn.Sequential(
            Conv2dBlock(
                in_channels=256,
                out_channels=256 * 32,
                activation=torch.nn.Identity(),
                conv_quantizer=Binarize(),
                kernel_size=3,
                groups=32,
            ),
            ChannelShuffle(groups=32),
            Conv2dBlock(
                in_channels=256 * 32,
                out_channels=256 * 4,
                activation=torch.nn.Identity(),
                conv_quantizer=Binarize(),
                kernel_size=3,
                groups=256 * 4,
            ),
            Conv2dBlock(
                in_channels=256 * 4,
                out_channels=256,
                activation=torch.nn.Identity(),
                conv_quantizer=Binarize(),
                kernel_size=3,
                groups=128,
            ),
        )
        self.compare_models_and_weight_shape(layers, expected)

    def test_generate_conv2d_sequence_with_width_more_than_2_last_groups_equal_out_channels(
        self,
    ):
        layers = generate_conv2d_sequence_with_width(
            in_channels=64,
            out_channels=256,
            activation=torch.nn.Identity(),
            weight_quantization=Binarize(),
            kernel_size=3,
            channel_width=8,
        )
        expected = torch.nn.Sequential(
            Conv2dBlock(
                in_channels=64,
                out_channels=256 * 8,
                activation=torch.nn.Identity(),
                conv_quantizer=Binarize(),
                kernel_size=3,
                groups=8,
            ),
            ChannelShuffle(groups=8),
            Conv2dBlock(
                in_channels=256 * 8,
                out_channels=256,
                activation=torch.nn.Identity(),
                conv_quantizer=Binarize(),
                kernel_size=3,
                groups=256,
            ),
        )
        self.compare_models_and_weight_shape(layers, expected)

    def test_pointwise_breakdown_forward(self):
        layer = depthwisePointwiseBreakdownConv2dBlock(
            in_channels=4,
            out_channels=8,
            pointwise_channel_width=2,
            kernel_size=3,
            activation=Binarize(),
            pointwise_activation=Binarize(),
        )
        test_input = torch.ones((2, 4, 3, 3))
        output = layer(test_input)
        self.assertSequenceEqual(output.shape, [2, 8, 1, 1])

    def test_BreakdownConv2d_block(self):
        layers = BreakdownConv2dBlock(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            groups=4,
            activation=torch.nn.Identity(),
            pointwise_activation=torch.nn.Identity(),
            conv_quantizer=None,
            pointwise_quantizer=None,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            padding_mode="zeros",
        )
        expected = torch.nn.Sequential(
            Conv2dBlock(
                in_channels=32,
                out_channels=16 * 4 * 2,
                activation=torch.nn.Identity(),
                conv_quantizer=None,
                kernel_size=2,
                groups=4,
            ),
            ChannelShuffle(groups=4),
            Conv2dBlock(
                in_channels=16 * 4 * 2,
                out_channels=16,
                activation=torch.nn.Identity(),
                conv_quantizer=None,
                kernel_size=1,
                groups=16,
            ),
        )
        mask = FixedOffsetMask4d(
            out_channels=16 * 4 * 2,
            in_channels=32,
            kernel_size=2,
            offset_axis=2,
            axis_width=1,
            groups=4,
        )
        parametrize.register_parametrization(expected[0].conv2d, "weight", mask)
        self.compare_models_and_weight_shape(
            list(layers.modules())[1:], list(expected.modules())[1:]
        )


if __name__ == "__main__":
    unittest.main()
