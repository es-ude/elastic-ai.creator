import random
import unittest

import brevitas.nn as bnn
import numpy as np
import torch
from brevitas import quant
from torch import nn

import elasticai.creator.brevitas.brevitas_quantizers as bquant
from elasticai.creator.brevitas.brevitas_model_comparison import (
    BrevitasModelComparisonTestCase,
)


class TestModelComparison(BrevitasModelComparisonTestCase):
    """
    Test the brevitas model comparison class
    """

    def setUp(self) -> None:
        self.ensure_reproducibility()

    @staticmethod
    def ensure_reproducibility():
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    @classmethod
    def modelA(cls):
        l1 = bnn.QuantLinear(in_features=5, out_features=2, bias=True)
        # we think brevitas is manipulating some seed therefore we need to reset them again after calling quantconv1d
        cls.ensure_reproducibility()
        return nn.Sequential(l1)

    @staticmethod
    def modelB():
        l1 = bnn.QuantConv1d(in_channels=5, out_channels=2, kernel_size=3)
        return nn.Sequential(l1)

    @staticmethod
    def modelC():
        l1 = bnn.QuantLinear(in_features=5, out_features=2, bias=True)
        l2 = bnn.QuantConv1d(in_channels=1, out_channels=1, kernel_size=2)
        return nn.Sequential(l1, l2)

    @classmethod
    def big_model(cls):
        l1 = bnn.QuantConv1d(
            in_channels=3,
            out_channels=3,
            kernel_size=2,
            weight_quant=bquant.BinaryWeights,
            bias_quant=bquant.BinaryBias,
        )
        l2 = nn.BatchNorm1d(num_features=3)
        l3 = bnn.QuantIdentity(act_quant=bquant.BinaryActivation)
        l4 = nn.MaxPool1d(
            kernel_size=3,
            stride=2,
        )
        l5 = nn.Flatten()
        l6 = bnn.QuantLinear(
            in_features=21, out_features=1, bias=True, weight_quant=bquant.BinaryWeights
        )
        l7 = nn.Sigmoid()

        # we think brevitas is manipulating some seed therefore we need to reset them again after calling quantconv1d
        cls.ensure_reproducibility()

        return nn.Sequential(l1, l2, l3, l4, l5, l6, l7)

    def test_same_easy_model(self):
        self.assertModelEqual(self.modelA(), self.modelA())

    def test_same_big_model(self):
        self.assertModelEqual(self.big_model(), self.big_model())

    def test_different_model_with_same_length(self):
        with self.assertRaises(AssertionError):
            self.assertModelEqual(self.modelA(), self.modelB())

    def test_different_model_with_different_length(self):
        with self.assertRaises(AssertionError):
            self.assertModelEqual(self.modelA(), self.modelC())

    def test_model_with_different_quantizers(self):
        model1 = nn.Sequential(
            bnn.QuantLinear(
                in_features=5,
                out_features=2,
                bias=True,
            )
        )

        model2 = nn.Sequential(
            bnn.QuantLinear(
                in_features=5,
                out_features=2,
                bias=True,
                weight_quant=quant.SignedBinaryWeightPerTensorConst,
            )
        )

        with self.assertRaises(AssertionError):
            self.assertModelEqual(model1, model2)


if __name__ == "__main__":
    unittest.main()
