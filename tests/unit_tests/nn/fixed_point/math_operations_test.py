import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.nn.fixed_point.math_operations import MathOperations
from tests.tensor_test_case import TensorTestCase


class FixedPointMathOperationsTest(TensorTestCase):
    def setUp(self) -> None:
        self.operations = MathOperations(
            config=FxpArithmetic(FxpParams(total_bits=4, frac_bits=2, signed=True))
        )

    def test_add(self) -> None:
        a = torch.tensor([-0.25, 0.5, 1.0])
        b = torch.tensor([-1.5, 1.0, 1.5])
        actual = self.operations.add(a, b)
        expected = [-1.75, 1.5, 1.75]
        self.assertTensorEqual(expected, actual)

    def test_matmul(self) -> None:
        a = torch.tensor([[-2.0, -1.75, -1.5], [-0.25, 0.0, 0.25], [1.25, 1.5, 1.75]])
        b = torch.tensor([[-0.25], [0.5], [0.25]])
        actual = self.operations.matmul(a, b)
        expected = [[-0.75], [0.0], [0.75]]
        self.assertTensorEqual(expected, actual)

    def test_mul(self) -> None:
        a = torch.tensor([-0.5, 1.5, 0.5])
        b = torch.tensor([0.5, 1.5, 1.2])
        actual = self.operations.mul(a, b)
        expected = [-0.25, 1.75, 0.5]
        self.assertTensorEqual(expected, actual)

    def test_quantize_clamps_minus5_to_minus2(self) -> None:
        a = torch.tensor([-5.0])
        actual = self.operations.quantize(a)
        expected = [-2.0]
        self.assertTensorEqual(expected, actual)

    def test_quantize_rounds_1_1_to_1_0(self) -> None:
        a = torch.tensor([1.1])
        actual = self.operations.quantize(a)
        expected = [1.0]
        self.assertTensorEqual(expected, actual)

    def test_round(self) -> None:
        a = torch.tensor([-0.5, 1.5, 0.5, 0.65])
        actual = self.operations.round(a)
        expected = [-0.5, 1.5, 0.5, 0.75]
        self.assertTensorEqual(expected, actual)

    def test_quantize(self) -> None:
        a = torch.tensor([-0.5, 1.5, 0.5, 0.65])
        actual = self.operations.quantize(a)
        expected = [-0.5, 1.5, 0.5, 0.5]
        self.assertTensorEqual(expected, actual)
