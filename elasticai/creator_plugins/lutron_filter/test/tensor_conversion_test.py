from collections import namedtuple
from unittest import TestCase

import torch

from elasticai.creator_plugins.lutron_filter.torch.tensor_conversion import (
    lutron_to_torch1d,
    torch1d_input_tensor_to_grouped_strings,
    torch1d_to_lutron,
)


class ConvertingTorchToLutron1dTensorFormatTest(TestCase):
    """
    Torch Format is a tensor with the following dimensions (B, C, N)
    where B is batch size, C is number of channels and N is the number of
    spatial steps.

    Our Lutron format is (B, CxN) where B is batch size. We use flat vectors
    of size CxN that provide data with interleaved channels, e.g., (0, 1, 2, 3)
    with two channels would mean that we have two spatial steps and the first channel
    consists of the data (0, 2) while the second channel has the data (1, 3).

    We want to
    - convert our format to numeric strings
      - [0, 1, 2] -> "012"
    - convert bit strings to our format
      - "012" -> [0, 1, 2]
    - convert our format to torch 1d format
      - [0, 1, 2, 3, 4, 5] -> [[0, 3], [1, 4], [2, 5]]
    - divide the torch 1d format into groups
      - [[0], [1], [2], [3]] -> [[[0], [1]], [[2], [3]]]
      - [[0, 4], [1, 5], [2, 6], [3, 7]] -> [[[0, 4], [1, 5]], [[2, 6], [3, 7]]]
    """

    def test_2x2_torch_to_lutron(self):
        x = torch.tensor([[0, 2], [1, 3]])
        expected = torch.tensor([0, 1, 2, 3])
        actual = torch1d_to_lutron(x)
        self.assertEqual(expected.tolist(), actual.tolist())

    def test_2x3_torch_to_lutron(self):
        x = torch.tensor([[0, 2, 4], [1, 3, 5]])
        expected = torch.tensor([0, 1, 2, 3, 4, 5])
        actual = torch1d_to_lutron(x)
        self.assertEqual(expected.tolist(), actual.tolist())

    def test_batched_2x2x3_torch_to_lutron(self):
        x = torch.tensor([[[0, 2, 4], [1, 3, 5]], [[6, 8, 10], [7, 9, 11]]])
        expected = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        actual = torch1d_to_lutron(x)
        self.assertEqual(expected.tolist(), actual.tolist())

    def test_3x2_torch_to_lutron(self):
        x = torch.tensor([[0, 3], [1, 4], [2, 5]])
        expected = torch.tensor([0, 1, 2, 3, 4, 5])
        actual = torch1d_to_lutron(x)
        self.assertEqual(expected.tolist(), actual.tolist())

    def test_lutron_to_torch_3x2(self):
        expected = torch.tensor([[0, 3], [1, 4], [2, 5]])
        x = torch.tensor([0, 1, 2, 3, 4, 5])
        actual = lutron_to_torch1d(x, channels=3)
        self.assertEqual(expected.tolist(), actual.tolist())

    def test_lutron_to_torch_3x2_batched(self):
        expected = torch.tensor([[[0, 3], [1, 4], [2, 5]], [[6, 9], [7, 10], [8, 11]]])
        x = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        actual = lutron_to_torch1d(x, channels=3)
        self.assertEqual(expected.tolist(), actual.tolist())


class IOTensor1DToBitStringListConverterTest(TestCase):
    Sample = namedtuple("Sample", ("channel_a", "channel_b", "channel_c", "channel_d"))
    Batch = namedtuple("Batch", ("sample_I", "sample_II"))

    def setUp(self):
        self.data = self.Batch(
            sample_I=self.Sample(
                channel_a=(0, 1), channel_b=(1, 0), channel_c=(0, 0), channel_d=(1, 1)
            ),
            sample_II=self.Sample(
                channel_a=(1, 0), channel_b=(0, 1), channel_c=(1, 1), channel_d=(0, 0)
            ),
        )
        self.channel_ab_sample_I = "0110"
        self.channel_ab_sample_II = "1001"
        self.channel_cd_sample_I = "0011"
        self.channel_cd_sample_II = "1100"

    def test_single_sample(self):
        x = torch.tensor([self.data[0]])
        group_0 = [self.channel_ab_sample_I]
        group_1 = [self.channel_cd_sample_I]
        expected = [group_0, group_1]
        actual = torch1d_input_tensor_to_grouped_strings(x, groups=2)
        self.assertEqual(expected, actual)

    def test_two_samples(self):
        x = torch.tensor(self.data)
        expected = [
            [self.channel_ab_sample_I, self.channel_ab_sample_II],
            [self.channel_cd_sample_I, self.channel_cd_sample_II],
        ]
        actual = torch1d_input_tensor_to_grouped_strings(x, groups=2)
        self.assertEqual(expected, actual)


class ConvertingBatchOfFlatTensorsToIOStringsWithSingleGroup(TestCase):
    Batch = namedtuple("Batch", ("sample_I", "sample_II"))

    def setUp(self):
        self.data = self.Batch(
            sample_I=(0, 0, 0, 1),
            sample_II=(1, 1, 1, 0),
        )
        self.sample_I = "0001"
        self.sample_II = "1110"

    def test_single_sample(self):
        x = torch.tensor([self.data[0]])
        expected = [[self.sample_I]]
        actual = torch1d_input_tensor_to_grouped_strings(x, groups=1)
        self.assertEqual(expected, actual)

    def test_two_samples(self):
        x = torch.tensor(self.data)
        expected = [
            [self.sample_I, self.sample_II],
        ]
        actual = torch1d_input_tensor_to_grouped_strings(x, groups=1)
        self.assertEqual(expected, actual)


class ConvertingBatchOfNumbersToIOStringsWithSingleGroup(TestCase):
    def test_convert_four_samples(self):
        x = torch.tensor([0, 1, 1, 0])
        expected = [["0", "1", "1", "0"]]
        actual = torch1d_input_tensor_to_grouped_strings(x, groups=1)
        self.assertEqual(expected, actual)
