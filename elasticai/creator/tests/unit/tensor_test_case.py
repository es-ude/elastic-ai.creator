from unittest import TestCase

from torch import Tensor


def _to_list(x: Tensor) -> list:
    return x.detach().numpy().tolist()


class TensorTestCase(TestCase):
    def assertTensorEqual(self, expected: list | Tensor, actual: list | Tensor) -> None:
        expected = _to_list(expected) if isinstance(expected, Tensor) else expected
        actual = _to_list(actual) if isinstance(actual, Tensor) else actual
        self.assertEqual(expected, actual)
