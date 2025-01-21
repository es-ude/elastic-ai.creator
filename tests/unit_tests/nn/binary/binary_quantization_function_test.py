from types import SimpleNamespace
from typing import cast

import torch
from torch import Tensor, tensor

from elasticai.creator.nn.binary.binary_quantization_function import Binarize
from tests.tensor_test_case import TensorTestCase


def binarize(x: Tensor) -> Tensor:
    return cast(Tensor, Binarize.apply(x))


class BinarizeFunctionTest(TensorTestCase):
    def test_ForwardRaisesErrorOnMissingInput(self) -> None:
        def save_for_backward_dummy(_):
            pass

        dummy_context = SimpleNamespace(save_for_backward=save_for_backward_dummy)
        try:
            Binarize.forward(dummy_context)
        except TypeError:
            return

        self.fail()

    def test_Yields1For0(self) -> None:
        self.assertTensorEqual(expected=tensor([1.0]), actual=binarize(tensor([0.0])))

    def test_Yields1For2Point4(self) -> None:
        self.assertTensorEqual(expected=tensor([1.0]), actual=binarize(tensor([2.4])))

    def test_YieldMinus1ForNegativeInput(self) -> None:
        self.assertTensorEqual(expected=tensor([-1.0]), actual=binarize(tensor([-2.8])))

    def check_gradient(self, expected_grad, x) -> None:
        x = torch.tensor([x], requires_grad=True)
        y = binarize(x)
        y.backward()
        self.assertTensorEqual(torch.tensor([expected_grad]), cast(Tensor, x.grad))

    def test_gradient_is_0_for_input_greater_1(self) -> None:
        self.check_gradient(expected_grad=0.0, x=1.1)

    def test_gradient_is_0_for_input_less_minus_1(self) -> None:
        self.check_gradient(expected_grad=0.0, x=-1.1)

    def test_gradient_is_1_for_input_between_minus_1_and_1(self) -> None:
        self.check_gradient(expected_grad=1.0, x=-0.8)
