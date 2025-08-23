import torch

from elasticai.creator.base_modules.math_operations_torch import TorchMathOperations
from tests.tensor_test_case import TensorTestCase


class TorchMathOperationsTest(TensorTestCase):
    def setUp(self) -> None:
        self.operations = TorchMathOperations()

    def test_quantize(self) -> None:
        a = torch.tensor([0, 1.5, 2.25, 3.123456])
        actual = self.operations.quantize(a)
        self.assertTensorEqual(a, actual)

    def test_add(self) -> None:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        actual = self.operations.add(a, b)
        expected = [5, 7, 9]
        self.assertTensorEqual(expected, actual)

    def test_mul(self) -> None:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        actual = self.operations.mul(a, b)
        expected = [4, 10, 18]
        self.assertTensorEqual(expected, actual)

    def test_matmul(self) -> None:
        a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = torch.tensor([[1], [2], [3]])
        actual = self.operations.matmul(a, b)
        expected = [[14], [32], [50]]
        self.assertTensorEqual(expected, actual)
