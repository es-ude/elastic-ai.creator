import unittest

import torch
from torch.nn.parameter import Parameter

from elasticai.creator.vhdl.custom_layers.linear import FixedPointLinear, _BaseLinear
from elasticai.creator.vhdl.number_representations import FixedPoint


class BaseLinearTest(unittest.TestCase):
    def test_default_linear(self) -> None:
        linear = _BaseLinear(in_features=3, out_features=1)
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias) * 4)

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = linear(input_tensor).detach().numpy().tolist()
        target = [10]

        self.assertEqual(actual, target)

    def test_linear_with_customized_add_op(self) -> None:
        linear = _BaseLinear(in_features=3, out_features=1, add_op=torch.sub)
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias) * 4)

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = linear(input_tensor).detach().numpy().tolist()
        target = [2]

        self.assertEqual(actual, target)

    def test_linear_with_customized_matmul_op(self) -> None:
        def custom_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b) * (-1)

        linear = _BaseLinear(in_features=3, out_features=1, matmul_op=custom_matmul)
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias) * 4)

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = linear(input_tensor).detach().numpy().tolist()
        target = [-2]

        self.assertEqual(actual, target)

    def test_linear_with_customized_add_op_and_matmul_op(self) -> None:
        def custom_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b) * (-1)

        linear = _BaseLinear(
            in_features=3, out_features=1, matmul_op=custom_matmul, add_op=torch.sub
        )
        linear.weight = Parameter(torch.ones_like(linear.weight))
        linear.bias = Parameter(torch.ones_like(linear.bias) * 4)

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = linear(input_tensor).detach().numpy().tolist()
        target = [-10]

        self.assertEqual(actual, target)

    def test_linear_with_customized_matmul_without_bias(self) -> None:
        def custom_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b) * (-1)

        linear = _BaseLinear(
            in_features=3, out_features=1, matmul_op=custom_matmul, bias=False
        )
        linear.weight = Parameter(torch.ones_like(linear.weight))

        input_tensor = torch.as_tensor([1, 2, 3], dtype=torch.float32)
        actual = linear(input_tensor).detach().numpy().tolist()
        target = [-6]

        self.assertEqual(actual, target)


class FixedPointLinearTest(unittest.TestCase):
    def test_fixed_point_linear_in_bounds(self) -> None:
        fp_factory = FixedPoint.get_factory(total_bits=16, frac_bits=8)
        linear = FixedPointLinear(
            in_features=3, out_features=1, fixed_point_factory=fp_factory
        )
        fp_one = int(fp_factory(1))
        linear.weight = Parameter(torch.ones_like(linear.weight) * fp_one)
        linear.bias = Parameter(torch.ones_like(linear.bias) * fp_one)

        input_tensor = torch.as_tensor(
            list(map(lambda x: fp_factory(x).to_signed_int(), [-7, 2, 3])),
            dtype=torch.float32,
        )
        actual = linear(input_tensor).detach().numpy().tolist()
        target = [fp_factory(-1).to_signed_int()]

        self.assertEqual(actual, target)

    def test_fixed_point_linear_out_of_bounds(self) -> None:
        fp_factory = FixedPoint.get_factory(total_bits=3, frac_bits=0)
        linear = FixedPointLinear(
            in_features=3, out_features=1, fixed_point_factory=fp_factory
        )
        fp_one = int(fp_factory(1))
        linear.weight = Parameter(torch.ones_like(linear.weight) * fp_one)
        linear.bias = Parameter(torch.ones_like(linear.bias) * fp_one)

        input_tensor = torch.as_tensor(
            list(map(lambda x: fp_factory(x).to_signed_int(), [1, 2, 3])),
            dtype=torch.float32,
        )
        actual = linear(input_tensor).detach().numpy().tolist()
        target = [7]

        self.assertEqual(actual, target)
