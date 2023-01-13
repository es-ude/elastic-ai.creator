from unittest import TestCase

from torch import Tensor


class TensorTestCase(TestCase):
    def assertTensorEquals(self, expected: Tensor, actual: Tensor):
        equals = expected.equal(actual)
        if not equals:
            raise AssertionError(
                "Tensors differ: {expected} != {actual}".format(
                    expected=expected, actual=actual
                )
            )
