import unittest

import torch

from elasticai.creator.nn.integer.math_operations.intadd import intadd


class TestIntAdd(unittest.TestCase):
    def test_valid_intadd(self):
        a_quant_bits = 8
        b_quant_bits = 8
        c_quant_bits = 9

        # creat a 3d tensor with random values of 8-bit quantization
        a = torch.tensor(
            [
                [[8, 107, -52, 71], [-31, 65, -99, -70], [113, -45, -75, 12]],
                [[-112, -113, 111, -80], [13, -76, 98, -110], [-79, -95, 118, 8]],
            ],
            dtype=torch.int32,
        )
        b = torch.tensor(
            [
                [[126, -121, 73, 5], [68, -128, 80, -73], [-28, -33, 65, -93]],
                [[-90, 53, -107, 80], [6, -125, -12, -38], [17, -113, -37, -6]],
            ],
            dtype=torch.int32,
        )

        c = intadd(a, b, c_quant_bits)
        expected_c = torch.tensor(
            [
                [[134, -14, 21, 76], [37, -63, -19, -143], [85, -78, -10, -81]],
                [[-202, -60, 4, 0], [19, -201, 86, -148], [-62, -208, 81, 2]],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(c, expected_c))
