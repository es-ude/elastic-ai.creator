import torch

from elasticai.creator.nn._two_complement_fixed_point_config import (
    TwoComplementFixedPointConfig,
)
from elasticai.creator.nn.fixed_point_arithmetics import FixedPointArithmetics
from elasticai.creator.nn.float_arithmetics import FloatArithmetics
from tests.tensor_test_case import TensorTestCase


class FloatArithmeticsTest(TensorTestCase):
    def setUp(self) -> None:
        self.ops = FloatArithmetics()

    def test_quantize(self) -> None:
        a = torch.tensor([0, 1.5, 2.25, 3.123456])
        actual = self.ops.quantize(a)
        self.assertTensorEqual(a, actual)

    def test_round(self) -> None:
        a = torch.tensor([0, 1.5, 2.25, 3.123456])
        actual = self.ops.round(a)
        self.assertTensorEqual(a, actual)

    def test_clamp(self) -> None:
        a = torch.tensor([0, 1.5, 2.25, 3.123456])
        actual = self.ops.clamp(a)
        self.assertTensorEqual(a, actual)

    def test_add(self) -> None:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        actual = self.ops.add(a, b)
        expected = [5, 7, 9]
        self.assertTensorEqual(expected, actual)

    def test_sum(self) -> None:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([-4, 5, -6])
        c = torch.tensor([1, 1, 1])
        actual = self.ops.sum(a, b, c)
        expected = [-2, 8, -2]
        self.assertTensorEqual(expected, actual)

    def test_mul(self) -> None:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        actual = self.ops.mul(a, b)
        expected = [4, 10, 18]
        self.assertTensorEqual(expected, actual)

    def test_matmul(self) -> None:
        a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = torch.tensor([[1], [2], [3]])
        actual = self.ops.matmul(a, b)
        expected = [[14], [32], [50]]
        self.assertTensorEqual(expected, actual)


class FixedPointArithmeticsTest(TensorTestCase):
    def setUp(self) -> None:
        self.config: TwoComplementFixedPointConfig = TwoComplementFixedPointConfig(
            total_bits=4, frac_bits=2
        )
        self.ops = FixedPointArithmetics(config=self.config)

    def test_quantize_clamps_minus5_to_minus2(self) -> None:
        a = torch.tensor([-5.0])
        actual = self.ops.quantize(a)
        expected = [-2.0]
        self.assertTensorEqual(expected, actual)

    def test_quantize_rounds_1_1_to_1_0(self) -> None:
        a = torch.tensor([1.1])
        actual = self.ops.quantize(a)
        expected = [1.0]
        self.assertTensorEqual(expected, actual)

    def test_clamp(self) -> None:
        a = torch.tensor([-5, -2.1, -1.0, 0.0, 1.8, 5])
        actual = self.ops.clamp(a)
        expected = [
            self.config.minimum_as_rational,
            self.config.minimum_as_rational,
            -1.0,
            0.0,
            self.config.maximum_as_rational,
            self.config.maximum_as_rational,
        ]
        self.assertTensorEqual(expected, actual)

    def test_round(self) -> None:
        a = torch.tensor([-2.0, -1.8, 1.25, 1.6])
        actual = self.ops.round(a)
        expected = [-2.0, -1.75, 1.25, 1.5]
        self.assertTensorEqual(expected, actual)

    def test_add(self) -> None:
        a = torch.tensor([-0.25, 0.5, 1.0])
        b = torch.tensor([-1.5, 1.0, 1.5])
        actual = self.ops.add(a, b)
        expected = [-1.75, 1.5, 1.75]
        self.assertTensorEqual(expected, actual)

    def test_sum(self) -> None:
        a = torch.tensor([-0.25, 0.5, 1.0])
        b = torch.tensor([-1.5, 1.0, 1.5])
        c = torch.tensor([-0.5, -1.0, -1.0])
        actual = self.ops.sum(a, b, c)
        expected = [-2.0, 0.5, 1.5]
        self.assertTensorEqual(expected, actual)

    def test_mul(self) -> None:
        a = torch.tensor([-0.5, 1.5, 0.5])
        b = torch.tensor([0.5, 1.5, 1.25])
        actual = self.ops.mul(a, b)
        expected = [-0.25, 1.75, 0.5]
        self.assertTensorEqual(expected, actual)

    def test_matmul(self) -> None:
        a = torch.tensor([[-2.0, -1.75, -1.5], [-0.25, 0.0, 0.25], [1.25, 1.5, 1.75]])
        b = torch.tensor([[-0.25], [0.5], [0.25]])
        actual = self.ops.matmul(a, b)
        expected = [[-0.75], [0.0], [0.75]]
        self.assertTensorEqual(expected, actual)