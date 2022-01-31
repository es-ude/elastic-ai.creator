import unittest

import torch

from elasticai.creator.breakdown import generate_conv2d_sequence_with_width,depthwisePointwiseBreakdownQConv2d_block
from elasticai.creator.layers import Binarize,ChannelShuffle
from elasticai.creator.blocks import Conv2d_block
class BreakdownTest(unittest.TestCase):
    def compare_models_and_weight_shape(self,expected,actual):
        self.assertTrue(len(expected) == len(actual), "number of layers")
        for layer_expected, layer_actual in zip(expected,actual):
            self.assertTrue( type(layer_actual) is type(layer_expected), f"layer types differ expected:{layer_expected} actual:{layer_actual}")
            if hasattr( layer_actual, 'conv2d'):
                self.assertSequenceEqual(layer_actual.conv2d.weight.shape,layer_expected.conv2d.weight.shape,f"layer shapes differ expected:{layer_expected.conv2d.weight.shape} actual:{layer_actual.conv2d.weight.shape}")
            if hasattr(layer_actual,'groups'):
                self.assertEqual(layer_actual.groups,layer_expected.groups,f"groups differ expected:{layer_expected.groups} actual:{layer_actual.groups}")
            
    
    def test_generate_conv2d_sequence_with_width_base(self):
        layers = generate_conv2d_sequence_with_width(in_channels=2,out_channels=4,activation=torch.nn.Identity(),weight_quantization=Binarize(),kernel_size=3,channel_width=2)
        expected = torch.nn.Sequential(Conv2d_block(in_channels=2,out_channels=4,activation=torch.nn.Identity(),conv_quantizer=Binarize(),kernel_size=3),ChannelShuffle(groups=1))
        self.compare_models_and_weight_shape(layers,expected)

    def test_generate_conv2d_sequence_with_width_more_complex(self):
        layers = generate_conv2d_sequence_with_width(in_channels=4,out_channels=2,activation=torch.nn.Identity(),weight_quantization=Binarize(),kernel_size=3,channel_width=2)
        expected = torch.nn.Sequential(Conv2d_block(in_channels=4,out_channels=4,activation=torch.nn.Identity(),conv_quantizer=Binarize(),kernel_size=3,groups=2),ChannelShuffle(groups=2),Conv2d_block(in_channels=4,out_channels=2,activation=torch.nn.Identity(),conv_quantizer=Binarize(),kernel_size=3,groups=2))
        self.compare_models_and_weight_shape(layers,expected)
        
    def test_generate_conv2d_sequence_with_width_more_than_2_last_groups_unequal_out_channels(self):
        layers = generate_conv2d_sequence_with_width(in_channels=256,out_channels=256,activation=torch.nn.Identity(),weight_quantization=Binarize(),kernel_size=3,channel_width=8)
        expected = torch.nn.Sequential(Conv2d_block(in_channels=256,out_channels=256*32,activation=torch.nn.Identity(),conv_quantizer=Binarize(),kernel_size=3,groups=32),ChannelShuffle(groups=32),Conv2d_block(in_channels=256*32,out_channels=256*4,activation=torch.nn.Identity(),conv_quantizer=Binarize(),kernel_size=3,groups=256*4),Conv2d_block(in_channels=256*4,out_channels=256,activation=torch.nn.Identity(),conv_quantizer=Binarize(),kernel_size=3,groups=128))
        self.compare_models_and_weight_shape(layers,expected)
    
    def test_generate_conv2d_sequence_with_width_more_than_2_last_groups_equal_out_channels(self):
        layers = generate_conv2d_sequence_with_width(in_channels=64,out_channels=256,activation=torch.nn.Identity(),weight_quantization=Binarize(),kernel_size=3,channel_width=8)
        expected = torch.nn.Sequential(Conv2d_block(in_channels=64,out_channels=256*8,activation=torch.nn.Identity(),conv_quantizer=Binarize(),kernel_size=3,groups=8),ChannelShuffle(groups=8),Conv2d_block(in_channels=256*8,out_channels=256,activation=torch.nn.Identity(),conv_quantizer=Binarize(),kernel_size=3,groups=256))
        self.compare_models_and_weight_shape(layers,expected)
    
    def test_pointwise_breakdown_forward(self):
        layer = depthwisePointwiseBreakdownQConv2d_block(in_channels=4,out_channels=8,pointwise_channel_width=2)

if __name__ == '__main__':
    unittest.main()
