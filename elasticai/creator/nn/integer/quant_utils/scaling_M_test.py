import unittest

import torch

from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M


class TestScalingM(unittest.TestCase):
    def test_scaling_M_basic(self):
        M = torch.tensor([0.12345], dtype=torch.float32)
        M_q_shift, M_q = scaling_M(M)

        self.assertTrue(M_q_shift.item() > 0)
        self.assertTrue(M_q_shift.item() <= 32)

        expected_M_q = torch.round(M * (2 ** M_q_shift.item())).type(torch.int32)
        torch.testing.assert_close(M_q, expected_M_q, rtol=1e-5, atol=1e-8)

        recovered_M = M_q * (2 ** (-M_q_shift.item()))
        error = (M - recovered_M) / M
        self.assertTrue(torch.all(error < 0.0001))
