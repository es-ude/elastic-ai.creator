from dataclasses import dataclass

import torch
from torch.nn.parameter import Parameter

from elasticai.creator.base_modules.linear import Linear, MathOperations
from elasticai.creator.base_modules.math_operations_torch import TorchMathOperations
from tests.tensor_test_case import TensorTestCase


def tensor(data: list) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float32)


def linear_base_with_fixed_params(
    in_features: int,
    out_features: int,
    bias: bool,
    operations: MathOperations = TorchMathOperations(),
) -> Linear:
    linear = Linear(
        in_features=in_features,
        out_features=out_features,
        operations=operations,
        bias=bias,
    )
    linear.weight = Parameter(torch.ones_like(linear.weight))
    if bias:
        linear.bias = Parameter(torch.ones_like(linear.bias))
    return linear


class AddThreeOperations(TorchMathOperations):
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return a + 3


@dataclass
class FixedMatmulResultOperations(TorchMathOperations):
    value: list[float]

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.value)


@dataclass
class FixedAddResultOperations(TorchMathOperations):
    value: list[float]

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.value)


class LinearTest(TensorTestCase):
    def test_with_bias(self) -> None:
        linear = linear_base_with_fixed_params(in_features=3, out_features=1, bias=True)

        actual = linear(tensor([1, 2, 3]))
        expected = [7.0]

        self.assertTensorEqual(expected, actual)

    def test_without_bias(self) -> None:
        linear = linear_base_with_fixed_params(
            in_features=3, out_features=1, bias=False
        )

        actual = linear(tensor([1, 2, 3]))
        expected = [6.0]

        self.assertTensorEqual(expected, actual)

    def test_different_quantization(self) -> None:
        linear = linear_base_with_fixed_params(
            in_features=3, out_features=1, bias=True, operations=AddThreeOperations()
        )

        actual = linear(tensor([1, 2, 3]))
        expected = [28.0]

        self.assertTensorEqual(expected, actual)

    def test_result_is_3_with_fixed_matmul(self) -> None:
        expected = [3.0]
        linear = linear_base_with_fixed_params(
            in_features=3,
            out_features=1,
            bias=False,
            operations=FixedMatmulResultOperations(expected),
        )

        actual = linear(tensor([1, 2, 3]))

        self.assertTensorEqual(expected, actual)

    def test_result_is_0_with_fixed_add(self) -> None:
        expected = [1.0]
        linear = linear_base_with_fixed_params(
            in_features=3,
            out_features=1,
            bias=True,
            operations=FixedAddResultOperations(expected),
        )

        actual = linear(tensor([1, 2, 3]))

        self.assertTensorEqual(expected, actual)
