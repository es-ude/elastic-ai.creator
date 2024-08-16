import logging
import unittest

import torch

from elasticai.creator.nn.integer.math_operations.subtraction import subtract
from elasticai.creator.nn.integer.quant_utils.QuantizedTensorValidator import (
    QuantizedTensorValidator,
)


class Testsubtract(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.ERROR)

    def test_subtract_int_8_correct(self):
        a = torch.tensor([127, 34, 56], dtype=torch.int32)
        b = torch.tensor([-54, 29, 39], dtype=torch.int32)
        c_quant_bits = 9

        results = subtract(a, b, c_quant_bits)
        expected = torch.tensor([127 + 54, 34 - 29, 56 - 39], dtype=torch.int32)

        torch.testing.assert_close(results, expected)

    def test_subtract_int_6_correct(self):
        a = torch.tensor([31, 5, -10], dtype=torch.int32)
        b = torch.tensor([-8, 29, 10], dtype=torch.int32)
        c_quant_bits = 7

        results = subtract(a, b, c_quant_bits)
        expected = torch.tensor([31 + 8, 5 - 29, -10 - 10], dtype=torch.int32)

        torch.testing.assert_close(results, expected)

    def test_subtract_int_4_correct(self):
        a = torch.tensor([7, 5, 0], dtype=torch.int32)
        b = torch.tensor([-8, 3, 10], dtype=torch.int32)
        c_quant_bits = 5

        results = subtract(a, b, c_quant_bits)
        expected = torch.tensor([7 + 8, 5 - 3, 0 - 10], dtype=torch.int32)

        torch.testing.assert_close(results, expected)

    def test_subtract_out_of_range(self):
        a = torch.tensor([127, 34, 56], dtype=torch.int32)
        b = torch.tensor([-54, 29, 39], dtype=torch.int32)
        c_quant_bits = 8

        with self.assertRaises(ValueError):
            subtract(a, b, c_quant_bits)
