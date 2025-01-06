from unittest import TestCase

import torch
from torch import Tensor, tensor

from elasticai.creator_plugins.lutron_filter.nn.lutron.truth_table_generation import (
    generate_input_tensor,
)


class InputTensorGenerationTest(TestCase):
    @staticmethod
    def create_expected_tensor(data: list[int]) -> Tensor:
        return tensor(data, dtype=torch.float16) * 2 - 1

    def test_single_group_and_channel_conv_enumerates_all_combinations(
        self,
    ) -> None:
        expected: Tensor = self.create_expected_tensor(
            [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
        )
        actual = generate_input_tensor(kernel_size=2, in_channels=1, groups=1)
        self.assertTrue(expected.equal(actual), msg="{} != {}".format(expected, actual))

    def test_two_channels_enumerates_along_correct_dim(self):
        expected = self.create_expected_tensor(
            [[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]]
        )
        actual = generate_input_tensor(kernel_size=1, in_channels=2, groups=1)
        self.assertTrue(expected.equal(actual), msg="{} != {}".format(expected, actual))

    def test_two_groups_repeats_values_along_channel_dim(self):
        expected = self.create_expected_tensor(
            [[[0, 0], [0, 0]], [[0, 1], [0, 1]], [[1, 0], [1, 0]], [[1, 1], [1, 1]]]
        )
        actual = generate_input_tensor(kernel_size=2, in_channels=2, groups=2)
        self.assertTrue(expected.equal(actual), msg="{} != {}".format(expected, actual))
