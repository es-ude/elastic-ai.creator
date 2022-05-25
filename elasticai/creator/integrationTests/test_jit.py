import pathlib
import re

import torch
from torch import jit
from torch.nn.functional import hardtanh

from elasticai.creator.qat.layers import Binarize
from elasticai.creator.tests.tensor_test_case import TensorTestCase


class JitTests(TensorTestCase):
    def test_script_a_sign_function(self):
        @jit.script
        def function(x):
            return x.sign()

        file_location = pathlib.Path(__file__).resolve()
        expected = f"""graph(%x.1 : Tensor):
  %2 : Tensor = aten::sign(%x.1)
  return (%2)
"""
        actual = f"{function.graph}"
        actual_with_comment_removed = re.sub(r" #(.*)\n", "\n", actual)
        self.assertEqual(expected, actual_with_comment_removed)

    def test_grad_of_clamp_is_zero_if_out_of_range(self):
        x = torch.tensor(-5.0, requires_grad=True)
        y = torch.clamp(x, -1.0, 1.0)
        y.backward()
        expected = torch.tensor(0.0)
        self.assertTensorEquals(expected, x.grad)

    def test_grad_of_clamp_is_one_if_in_range(self):
        x = torch.tensor(-0.5, requires_grad=True)
        y = torch.nn.functional.hardtanh(x)
        y.backward()
        expected = torch.tensor(1.0)
        self.assertTensorEquals(expected, x.grad)

    def test_grad_of_bin_ste(self):
        x = torch.tensor(0.3, requires_grad=True)
        y = Binarize()(x)
        y.backward()
        expected = torch.tensor(1.0)
        self.assertTensorEquals(expected, x.grad)

    def test_grad_of_bin_ste_is_zero(self):
        x = torch.tensor([1.2], requires_grad=True)
        y = Binarize()(x)
        y.backward()
        expected = torch.tensor([0.0])
        self.assertTensorEquals(expected, x.grad)

    def test_value_of_bin_ste(self):
        x = torch.tensor([0.3])
        y = Binarize()(x)
        expected = torch.tensor([1.0])
        self.assertTensorEquals(expected, y)

    def test_negative_value_of_bin_ste(self):
        x = torch.tensor([-0.3])
        y = Binarize()(x)
        expected = torch.tensor([-1.0])
        self.assertTensorEquals(expected, y)

    def test_script_heaviside_based_ste(self):
        function = jit.trace(Binarize(), torch.tensor([0.0]))
        actual = function(torch.tensor([-1.2, -0.4, 8.9]))
        expected = torch.tensor([-1.0, -1.0, 1.0])
        self.assertTensorEquals(expected, actual)
