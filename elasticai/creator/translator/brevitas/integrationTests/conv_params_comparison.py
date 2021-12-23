import unittest

import brevitas.nn as bnn


class ConvTest(unittest.TestCase):
    """
    self implemented Test case for comparing two brevitas convolutional layers
    """

    def assertConvParams(self, target, translated):
        self.assertEqual(target.in_channels, translated.in_channels)
        self.assertEqual(target.out_channels, translated.out_channels)
        self.assertEqual(target.kernel_size, translated.kernel_size)
        self.assertEqual(target.stride, translated.stride)
        self.assertEqual(target.padding, translated.padding)
        self.assertEqual(target.dilation, translated.dilation)
        self.assertEqual(target.groups, translated.groups)
        self.assertEqual(target.bias is not None, translated.bias is not None)
        self.assertEqual(target.padding_type, translated.padding_type)
        self.assertEqual(str(target.weight_quant), str(translated.weight_quant))
        self.assertEqual(str(target.bias_quant), str(translated.bias_quant))
        self.assertEqual(str(target), str(translated))

    def assertConv1dParams(self, target, translated):
        self.assertIsInstance(translated, bnn.QuantConv1d)
        self.assertConvParams(target, translated)

    def assertConv2dParams(self, target, translated):
        self.assertIsInstance(translated, bnn.QuantConv2d)
        self.assertConvParams(target, translated)
