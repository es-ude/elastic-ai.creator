from types import SimpleNamespace

import torch
from torch import Tensor

from elasticai.creator.qat.functional import binarize
from elasticai.creator.tests.tensor_test_case import TensorTestCase


class BinarizeFunctionTest(TensorTestCase):
    def test_ForwardRaisesErrorOnMissingInput(self):
        def save_for_backward_dummy(_):
            pass

        dummy_context = SimpleNamespace(save_for_backward=save_for_backward_dummy)
        try:
            binarize.forward(dummy_context)
        except TypeError:
            return

        self.fail()

    def test_Yields1For0(self):
        self.assertTensorEquals(
            expected=Tensor([1.0]), actual=binarize.apply(Tensor([0.0]))
        )

    def test_Yields1For2Point4(self):
        self.assertTensorEquals(
            expected=Tensor([1.0]), actual=binarize.apply(Tensor([2.4]))
        )

    def test_YieldMinus1ForNegativeInput(self):
        self.assertTensorEquals(
            expected=Tensor([-1.0]), actual=binarize.apply(Tensor([-2.8]))
        )

    def check_gradient(self, expected_grad, x):
        x = torch.tensor([x], requires_grad=True)
        y = binarize.apply(x)
        y.backward()
        self.assertTensorEquals(torch.tensor([expected_grad]), x.grad)

    def test_gradient_is_0_for_input_greater_1(self):
        self.check_gradient(expected_grad=0.0, x=1.1)

    def test_gradient_is_0_for_input_less_minus_1(self):
        self.check_gradient(expected_grad=0.0, x=-1.1)

    def test_gradient_is_1_for_input_between_minus_1_and_1(self):
        self.check_gradient(expected_grad=1.0, x=-0.8)
