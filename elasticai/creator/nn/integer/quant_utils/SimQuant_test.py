import unittest

import torch

from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant


class TestSimQuant(unittest.TestCase):
    def setUp(self):
        self.x_r = torch.tensor([0.5, -1.5, 3.0, 2.5, -0.5], dtype=torch.float32)
        self.grad_output = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

        self.min_quant = torch.tensor(-128, dtype=torch.int32)
        self.max_quant = torch.tensor(127, dtype=torch.int32)
        self.min_float = torch.tensor(-1.5, dtype=torch.float32)
        self.max_float = torch.tensor(3.0, dtype=torch.float32)
        self.scale_factor = torch.tensor(0.1, dtype=torch.float32)
        self.zero_point = torch.tensor(0, dtype=torch.int32)

        self.x_r_QParams = QParams(
            min_float=self.min_float,
            max_float=self.max_float,
            scale_factor=self.scale_factor,
            zero_point=self.zero_point,
        )

    def test_forward(self):
        # 执行前向传播
        result = QuantizeDequantizeFunction.apply(self.x_r, self.x_r_QParams)

        # 手动量化和反量化来验证结果
        expected_result = self.x_r_QParams.dequantize(
            self.x_r_QParams.quantize(self.x_r)
        )
        torch.testing.assert_close(result, expected_result, rtol=1e-5, atol=1e-8)

    def test_backward(self):
        # 执行前向传播
        result = QuantizeDequantizeFunction.apply(self.x_r, self.x_r_QParams)

        # 手动执行反向传播
        result.backward(self.grad_output)

        expected_grad_input = self.grad_output.clone()
        expected_grad_input[self.x_r.gt(self.max_float)] = 0
        expected_grad_input[self.x_r.lt(self.min_float)] = 0

        # 验证梯度
        grad_input = self.x_r.grad
        torch.testing.assert_close(
            grad_input, expected_grad_input, rtol=1e-5, atol=1e-8
        )
