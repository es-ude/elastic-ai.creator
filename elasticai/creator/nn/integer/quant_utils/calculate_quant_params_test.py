import unittest

import torch

from elasticai.creator.nn.integer.quant_utils.calculate_quant_params import (
    calculate_asymmetric_quant_params,
    calculate_symmetric_quant_params,
)


class TestCalculateQuantParams(unittest.TestCase):
    def test_calculate_asymmetric_quant_params_signed(self):
        min_float = torch.tensor([-3.2], dtype=torch.float32)
        max_float = torch.tensor([2.5], dtype=torch.float32)
        min_quant = torch.tensor([-128], dtype=torch.int32)
        max_quant = torch.tensor([127], dtype=torch.int32)
        eps = torch.tensor(torch.finfo(torch.float32).eps)

        scale_factor, zero_point, min_float, max_float = (
            calculate_asymmetric_quant_params(
                min_float, max_float, min_quant, max_quant, eps
            )
        )

        self.assertAlmostEqual(
            scale_factor, torch.tensor([0.0223529413], dtype=torch.float32), places=10
        )
        self.assertEqual(zero_point, torch.tensor([15], dtype=torch.int32))
        self.assertEqual(min_float, torch.tensor([-3.2], dtype=torch.float32))
        self.assertEqual(max_float, torch.tensor([2.5], dtype=torch.float32))

    def test_calculate_symmetric_quant_params_signed(self):
        min_quant = torch.tensor([-127], dtype=torch.int32)
        max_quant = torch.tensor([127], dtype=torch.int32)
        min_float = torch.tensor([-1.002], dtype=torch.float32)
        max_float = torch.tensor([1.0], dtype=torch.float32)
        eps = torch.tensor(torch.finfo(torch.float32).eps)

        scale_factor, zero_point, min_float, max_float = (
            calculate_symmetric_quant_params(
                min_quant, max_quant, min_float, max_float, eps
            )
        )

        self.assertAlmostEqual(
            scale_factor, torch.tensor([0.00788976377], dtype=torch.float32), places=6
        )
        self.assertEqual(zero_point, torch.tensor([0], dtype=torch.int32))
        self.assertEqual(min_float, torch.tensor([-1.002], dtype=torch.float32))
        self.assertEqual(max_float, torch.tensor([1.002], dtype=torch.float32))
