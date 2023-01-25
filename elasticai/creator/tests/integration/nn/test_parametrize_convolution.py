import unittest

import torch
from torch.nn import Conv1d
from torch.nn.utils.parametrize import register_parametrization

from elasticai.creator.nn.layers import Binarize


class ParametrizeConvolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def test_parametrize_conv1d(self) -> None:
        layer = Conv1d(in_channels=2, out_channels=3, kernel_size=(1,), bias=False)

        register_parametrization(layer, "weight", Binarize())

        inputs = torch.tensor([[[1], [1]]], dtype=torch.float32)
        outputs = layer(inputs).detach().numpy().tolist()

        self.assertEqual(type(layer.parametrizations["weight"][0]), Binarize)
        self.assertEqual(outputs, [[[2.0], [0.0], [0.0]]])
