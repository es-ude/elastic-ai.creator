import unittest

import torch

from elasticai.creator.nn.integer.math_operations.addition import add


class TestAddition(unittest.TestCase):
    def test_add_int_8_correct(self):
        a = torch.tensor([127, 34, 56], dtype=torch.int32)
        b = torch.tensor([-54, 29, 39], dtype=torch.int32)
        c_quant_bits = 9

        results = add(a, b, c_quant_bits)
        expected = torch.tensor([127 - 54, 34 + 29, 56 + 39], dtype=torch.int32)

        torch.testing.assert_close(results, expected)

    def test_add_int_6_correct(self):
        a = torch.tensor([31, 5, -10], dtype=torch.int32)
        b = torch.tensor([-8, 29, 10], dtype=torch.int32)
        c_quant_bits = 7

        results = add(a, b, c_quant_bits)
        expected = torch.tensor([31 - 8, 5 + 29, -10 + 10], dtype=torch.int32)

        torch.testing.assert_close(results, expected)

    def test_add_int_4_correct(self):
        a = torch.tensor([7, 5, 0], dtype=torch.int32)
        b = torch.tensor([-8, 3, 10], dtype=torch.int32)
        c_quant_bits = 5

        results = add(a, b, c_quant_bits)
        expected = torch.tensor([7 - 8, 5 + 3, 0 + 10], dtype=torch.int32)

        torch.testing.assert_close(results, expected)
