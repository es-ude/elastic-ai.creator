import unittest

import torch
from torch import nn

from elasticai.creator.qat.blocks import Conv1dBlock, DepthwiseConv1dBlock, LinearBlock
from elasticai.creator.qat.layers import Binarize, Ternarize


class BlockTests(unittest.TestCase):
    def test_conv1d_block(self):
        with self.subTest("test params"):
            module = Conv1dBlock(
                in_channels=2,
                out_channels=32,
                kernel_size=3,
                conv_quantizer=Binarize(),
                activation=Ternarize(),
                stride=1,
                padding=0,
                dilation=1,
                groups=2,
                bias=True,
                padding_mode="zeros",
                constraints=None,
            )
            self.assertEqual(module.conv1d.groups, 2)
            self.assertEqual(module.batch_norm.num_features, 32)

        with self.subTest("test call"):
            module = Conv1dBlock(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                conv_quantizer=Binarize(),
                activation=Ternarize(),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
                constraints=None,
            )
            module.eval()
            input = torch.randn([1, 1, 32])
            out = module(input)
            self.assertEqual(list(out.size()), [1, 32, 30])

    def test_Depthwiseconv1d_block(self):
        with self.subTest("test params"):
            module = DepthwiseConv1dBlock(
                in_channels=4,
                out_channels=32,
                kernel_size=3,
                conv_quantizer=Binarize(),
                activation=Ternarize(),
                stride=1,
                padding=0,
                dilation=1,
                bias=True,
                padding_mode="zeros",
                constraints=None,
                pointwise_quantizer=Binarize(),
                pointwise_activation=Binarize(),
            )
            self.assertEqual(module.depthwiseConv1d.groups, 4)
            self.assertEqual(module.batchnorm.num_features, 4)
            self.assertEqual(module.pointwise_batchnorm.num_features, 32)

        with self.subTest("test call"):
            module = DepthwiseConv1dBlock(
                in_channels=2,
                out_channels=32,
                kernel_size=3,
                conv_quantizer=Binarize(),
                activation=Ternarize(),
                stride=1,
                padding=0,
                dilation=1,
                bias=True,
                padding_mode="zeros",
                constraints=None,
                pointwise_quantizer=Binarize(),
                pointwise_activation=Binarize(),
            )
            module.eval()
            input = torch.randn([1, 2, 32])
            out = module(input)
            self.assertEqual(list(out.size()), [1, 32, 30])

    def test_Linear_block(self):
        with self.subTest("test params"):
            module = LinearBlock(
                in_features=16,
                out_features=4,
                linear_quantizer=Binarize(),
                activation=Ternarize(),
                bias=True,
            )
            self.assertEqual(module.linear.out_features, 4)
            self.assertEqual(module.batchnorm.num_features, 4)

        with self.subTest("test call"):
            module = LinearBlock(
                in_features=16,
                out_features=4,
                linear_quantizer=Binarize(),
                activation=nn.Identity(),
                bias=True,
            )
            module.eval()
            input = torch.randn([1, 16])
            out = module(input)
            self.assertEqual(list(out.size()), [1, 4])


if __name__ == "__main__":
    unittest.main()
