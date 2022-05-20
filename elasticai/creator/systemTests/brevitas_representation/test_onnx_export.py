import unittest

import brevitas.nn as bnn
from torch import nn

import elasticai.creator.brevitas.brevitas_quantizers as bquant
from elasticai.creator.brevitas.brevitas_model_comparison import (
    BrevitasModelComparisonTestCase,
)
from elasticai.creator.brevitas.brevitas_representation import BrevitasRepresentation
from elasticai.creator.systemTests.brevitas_representation.models_definition import (
    create_brevitas_model,
)


def create_small_brevitas_model():
    model = nn.Sequential(
        bnn.QuantConv1d(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            bias=False,
            weight_quant=bquant.BinaryWeights,
        ),
        nn.BatchNorm1d(num_features=3),
        bnn.QuantIdentity(act_quant=bquant.BinaryActivation),
        nn.MaxPool1d(kernel_size=5, stride=3),
        nn.Flatten(),
        bnn.QuantLinear(
            in_features=126,
            out_features=6,
            bias=False,
            weight_quant=bquant.BinaryWeights,
        ),
        nn.Sigmoid(),
    )
    return model


class SaveBrevitasModelToOnnx(BrevitasModelComparisonTestCase):
    """
    train a brevitas model and try to save it to onnx
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.brevitas_model = create_small_brevitas_model()
        cls.bigger_brevitas_model = create_brevitas_model()

    def test_save_to_onnx(self) -> None:
        br = BrevitasRepresentation(
            original_model=None, translated_model=self.brevitas_model
        )
        br.save_to_onnx((3, 1, 132), path="brevitas_model.onnx")

    def test_bigger_model_save_to_onnx(self) -> None:
        br = BrevitasRepresentation(
            original_model=None, translated_model=self.bigger_brevitas_model
        )
        br.save_to_onnx((3, 1, 97), path="bigger_brevitas_model.onnx")


if __name__ == "__main__":
    unittest.main()
