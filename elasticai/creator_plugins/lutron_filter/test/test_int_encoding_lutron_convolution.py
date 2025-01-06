import pytest
import torch
from torch.nn import Conv1d

from elasticai.creator_plugins.lutron_filter.nn.lutron.binarize import Binarize
from elasticai.creator_plugins.lutron_filter.torch.lutron_modules import (
    IntEncodingLutronConv1d,
)
from elasticai.creator_plugins.lutron_filter.torch.lutron_modules.int_encoding_lutron_convolution import (
    convert_lutron_bits_to_ints_as_two_complements,
)


@pytest.fixture
def identity_conv() -> Conv1d:
    c = Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=False)
    c.weight.data = torch.tensor([[[1.0]]])
    return c


@pytest.fixture
def encoding_conv_2bit(identity_conv) -> IntEncodingLutronConv1d:
    c = IntEncodingLutronConv1d(wrapped=identity_conv, bits=2, binarize=Binarize())
    c.eval()
    return c


class TestConvertingLutronBitVectorsToInts:
    def test_1_0_1_to_5(self):
        numbers = [[[5.0]]]
        lutron_rep = torch.tensor([[[-1], [-1], [1], [-1], [1]]], dtype=torch.float32)
        expected = torch.tensor(numbers, dtype=torch.float32)
        assert expected == convert_lutron_bits_to_ints_as_two_complements(lutron_rep)

    def test_1_0_1_to_minus_3_with_twos_complement(self):
        numbers = [[[-3.0]]]
        lutron_rep = torch.tensor([[[1], [-1], [1]]], dtype=torch.float32)
        expected = torch.tensor(numbers, dtype=torch.float32)
        assert expected == convert_lutron_bits_to_ints_as_two_complements(lutron_rep)

    def test_101_and_110_to_5_and_6(self):
        expected = torch.tensor([[[5.0, 6.0]]], dtype=torch.float32)
        lutron_rep = torch.tensor(
            [[[-1, -1], [1, 1], [-1, 1], [1, -1]]], dtype=torch.float32
        )
        assert (
            expected.tolist()
            == convert_lutron_bits_to_ints_as_two_complements(lutron_rep).tolist()
        )

    def test_conversion_for_two_channels(self):
        expected = torch.tensor([[[5.0], [6.0]]])
        lutron_rep = torch.tensor(
            [[[-1.0], [1.0], [-1.0], [1.0], [-1.0], [1.0], [1.0], [-1.0]]]
        )
        assert (
            expected.tolist()
            == convert_lutron_bits_to_ints_as_two_complements(
                lutron_rep, in_channels=2
            ).tolist()
        )


class TestInferenceWithTorchNumbers:
    def test_inference(self, encoding_conv_2bit: IntEncodingLutronConv1d) -> None:
        c = encoding_conv_2bit
        x = torch.tensor([[[1.0]]])
        y = c(x)
        assert x.tolist() == y.tolist()
