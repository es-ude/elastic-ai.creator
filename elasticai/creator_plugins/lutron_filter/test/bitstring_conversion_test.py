from unittest import TestCase

import torch

from elasticai.creator_plugins.lutron_filter.nn.lutron.truth_table_generation import (
    convert_to_list,
)


class ConvertGroupedTensorsForConvsTest(TestCase):
    def test_0(self):
        expected = [[(x, "0") for x in ["00", "01", "10", "11"]] for _ in range(2)]
        groups = 2
        inputs = torch.tensor(
            [
                [[0], [0]] * groups,
                [[0], [1]] * groups,
                [[1], [0]] * groups,
                [[1], [1]] * groups,
            ]
        )
        outputs = torch.zeros(4, groups, 1, dtype=torch.int)
        actual = convert_to_list(inputs, outputs, groups=2)
        self.assertEqual(expected, actual)

    def test_1(self):
        out_channels = 3
        expected = [[(x, "0") for x in ["0", "1"]] for _ in range(out_channels)]
        groups = 3
        inputs = torch.tensor(
            [
                [[0]] * groups,
                [[1]] * groups,
            ]
        )
        outputs = torch.zeros(2, groups, 1, dtype=torch.int)
        actual = convert_to_list(inputs, outputs, groups=3)
        self.assertEqual(expected, actual)
