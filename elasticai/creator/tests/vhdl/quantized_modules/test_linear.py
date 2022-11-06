import unittest

import torch
from torch.nn.parameter import Parameter

from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.quantized_modules.linear import (
    FixedPointLinear,
    _LinearBase,
)


def to_list(x: torch.Tensor) -> list[float]:
    return x.detach().numpy().tolist()


class LinearBaseTest(unittest.TestCase):
    def test_default_linear(self) -> None:
        linear = _LinearBase(in_features=3, out_features=1)
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias) * 4)

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = to_list(linear(input_tensor))
        target = [10]

        self.assertEqual(actual, target)

    def test_linear_with_customized_add_op(self) -> None:
        linear = _LinearBase(in_features=3, out_features=1, add_op=torch.sub)
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias) * 4)

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = to_list(linear(input_tensor))
        target = [2]

        self.assertEqual(actual, target)

    def test_linear_with_customized_matmul_op(self) -> None:
        def custom_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b) * (-1)

        linear = _LinearBase(in_features=3, out_features=1, matmul_op=custom_matmul)
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias) * 4)

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = to_list(linear(input_tensor))
        target = [-2]

        self.assertEqual(actual, target)

    def test_linear_with_customized_add_op_and_matmul_op(self) -> None:
        def custom_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b) * (-1)

        linear = _LinearBase(
            in_features=3, out_features=1, matmul_op=custom_matmul, add_op=torch.sub
        )
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias) * 4)

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = to_list(linear(input_tensor))
        target = [-10]

        self.assertEqual(actual, target)

    def test_linear_with_customized_matmul_without_bias(self) -> None:
        def custom_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b) * (-1)

        linear = _LinearBase(
            in_features=3, out_features=1, matmul_op=custom_matmul, bias=False
        )
        linear.weight = Parameter(torch.ones_like(linear.weight))

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = to_list(linear(input_tensor))
        target = [-6]

        self.assertEqual(actual, target)

    def test_linear_with_input_quant_and_input_dequant(self) -> None:
        linear = _LinearBase(
            in_features=3,
            out_features=1,
            input_quant=lambda x: x + 5,
            input_dequant=lambda x: x - 2,
        )
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias))

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = to_list(linear(input_tensor))
        target = [16]

        self.assertEqual(actual, target)

    def test_linear_with_param_quant_and_param_dequant(self) -> None:
        linear = _LinearBase(
            in_features=3,
            out_features=1,
            param_quant=lambda x: x + 5,
            param_dequant=lambda x: x - 2,
        )
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias))

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = to_list(linear(input_tensor))
        target = [28]

        self.assertEqual(actual, target)

    def test_linear_with_input_param_quant_dequant(self) -> None:
        linear = _LinearBase(
            in_features=3,
            out_features=1,
            input_quant=lambda x: x + 3,
            input_dequant=lambda x: x - 2,
            param_quant=lambda x: x + 5,
            param_dequant=lambda x: x - 2,
        )
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias))

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = to_list(linear(input_tensor))
        target = [40]

        self.assertEqual(actual, target)

    def test_linear_quantized_forward_raises_error(self) -> None:
        with self.assertRaises(NotImplementedError):
            _LinearBase(1, 2).quantized_forward(torch.ones(1))


class FixedPointLinearTest(unittest.TestCase):
    def test_fixed_point_linear_in_bounds(self) -> None:
        fp_factory = FixedPoint.get_factory(total_bits=16, frac_bits=8)
        linear = FixedPointLinear(
            in_features=3, out_features=1, fixed_point_factory=fp_factory
        )
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias))

        input_tensor = torch.as_tensor([-7, 2, 3], dtype=torch.float32)
        actual = to_list(linear(input_tensor))
        target = [-1]

        self.assertEqual(actual, target)

    def test_fixed_point_linear_out_of_bounds(self) -> None:
        fp_factory = FixedPoint.get_factory(total_bits=3, frac_bits=0)
        linear = FixedPointLinear(
            in_features=3, out_features=1, fixed_point_factory=fp_factory
        )
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias))

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = to_list(linear(input_tensor))
        target = [7]

        self.assertEqual(actual, target)

    def test_fixed_point_linear_quantized_forward(self) -> None:
        fp_factory = FixedPoint.get_factory(total_bits=8, frac_bits=4)
        linear = FixedPointLinear(
            in_features=3, out_features=1, fixed_point_factory=fp_factory
        )
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias))

        input_tensor = torch.tensor(
            [fp_factory(x).to_signed_int() for x in [-1, 0.5, 2]], dtype=torch.float32
        )
        expected = [40]
        actual = to_list(linear.quantized_forward(input_tensor))

        self.assertEquals(expected, actual)
