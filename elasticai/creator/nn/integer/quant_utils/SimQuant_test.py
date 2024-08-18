import unittest

import torch

from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant


class TestSimQuant(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([0.5, -1.5, 3.0, 2.5, -0.5], dtype=torch.float32)
        self.x_r = torch.tensor([0.5, -1.5, 3.0, 2.5, -0.5], dtype=torch.float32)
        self.grad_output = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

        self.min_quant = torch.tensor(-128, dtype=torch.int32)
        self.max_quant = torch.tensor(127, dtype=torch.int32)
        self.min_float = torch.tensor(-1.5, dtype=torch.float32)
        self.max_float = torch.tensor(3.0, dtype=torch.float32)

        self.scale_factor = torch.tensor([0.0176], dtype=torch.float32)
        self.zero_point = torch.tensor([-85], dtype=torch.int32)

        self.x_r_QParams = AsymmetricSignedQParams(
            quant_bits=8, observer=GlobalMinMaxObserver
        )
        self.x_r_QParams.min_float = self.min_float
        self.x_r_QParams.max_float = self.max_float
        self.x_r_QParams.scale_factor = self.scale_factor
        self.x_r_QParams.zero_point = self.zero_point

    def test_forward(self):
        result = SimQuant.apply(self.x_r, self.x_r_QParams)

        expected_result = self.x_r_QParams.dequantize(
            self.x_r_QParams.quantize(self.x_r)
        )
        torch.testing.assert_close(result, expected_result, rtol=1e-5, atol=1e-8)

    def test_backward(self):
        x_r = self.x_r.clone().requires_grad_(True)
        result = SimQuant.apply(x_r, self.x_r_QParams)
        result.backward(self.grad_output)

        expected_result = self.grad_output.clone()
        expected_result[self.x_r.gt(self.max_float)] = 0
        expected_result[self.x_r.lt(self.min_float)] = 0
        torch.testing.assert_close(x_r.grad, expected_result, rtol=1e-5, atol=1e-8)
