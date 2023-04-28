import torch

from elasticai.creator.base_modules.float_arithmetics import FloatArithmetics
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
