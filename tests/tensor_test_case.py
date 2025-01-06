from unittest import TestCase

from torch import Tensor


def _unify_inputs(*inputs: float | list | Tensor) -> list[float | list]:
    def unify(obj: float | list | Tensor) -> float | list:
        if isinstance(obj, (float, list)):
            return obj
        return obj.detach().tolist()

    return list(map(unify, inputs))


class TensorTestCase(TestCase):
    def assertTensorEqual(
        self, expected: float | list | Tensor, actual: float | list | Tensor
    ) -> None:
        self.assertEqual(*_unify_inputs(expected, actual))


def assertTensorEqual(
    expected: float | list | Tensor, actual: float | list | Tensor
) -> None:
    expected, actual = _unify_inputs(expected, actual)
    assert expected == actual
