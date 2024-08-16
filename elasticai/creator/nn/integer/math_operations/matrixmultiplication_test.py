import unittest

import torch

from elasticai.creator.nn.integer.math_operations.matrixmultiplication import matmul


class TestMatrixMultiplication(unittest.TestCase):
    def test_matmul_int_8_correct(self):
        a = torch.tensor([[127, 34], [56, 78]], dtype=torch.int32)
        b = torch.tensor([[-54, 29], [39, 12]], dtype=torch.int32)
        c_quant_bits = 18

        results = matmul(a, b, c_quant_bits)
        expected = torch.tensor(
            [
                [127 * -54 + 34 * 39, 127 * 29 + 34 * 12],
                [56 * -54 + 78 * 39, 56 * 29 + 78 * 12],
            ],
            dtype=torch.int32,
        )

        torch.testing.assert_close(results, expected)

    def test_matmul_int_6_correct(self):
        a = torch.tensor([[31, 5], [-10, 6]], dtype=torch.int32)
        b = torch.tensor([[-8, 29], [10, 12]], dtype=torch.int32)
        c_quant_bits = 14

        results = matmul(a, b, c_quant_bits)
        expected = torch.tensor(
            [
                [31 * -8 + 5 * 10, 31 * 29 + 5 * 12],
                [-10 * -8 + 6 * 10, -10 * 29 + 6 * 12],
            ],
            dtype=torch.int32,
        )

        torch.testing.assert_close(results, expected)

    def test_matmul_int_4_correct(self):
        a = torch.tensor([[7, 5], [0, 3]], dtype=torch.int32)
        b = torch.tensor([[-8, 3], [10, 12]], dtype=torch.int32)
        c_quant_bits = 10

        results = matmul(a, b, c_quant_bits)
        expected = torch.tensor(
            [
                [7 * -8 + 5 * 10, 7 * 3 + 5 * 12],
                [0 * -8 + 3 * 10, 0 * 3 + 3 * 12],
            ],
            dtype=torch.int32,
        )

        torch.testing.assert_close(results, expected)
