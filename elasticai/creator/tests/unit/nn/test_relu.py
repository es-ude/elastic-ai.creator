import unittest

import torch
import torch.nn.functional as F

from elasticai.creator.nn.relu import FixedPointReLU, _ReLUBase
from elasticai.creator.tests.unit.nn.utils import (
    from_fixed_point,
    to_fixed_point,
    to_list,
)
from elasticai.creator.vhdl.number_representations import FixedPoint


class ReLUBaseTest(unittest.TestCase):
    def test_relu_base_not_quantized_behaves_equivalent_to_relu(self) -> None:
        relu = _ReLUBase()

        xs = torch.linspace(-10, 10, 100)
        expected = to_list(F.relu(xs))
        actual = to_list(relu(xs))

        self.assertEqual(expected, actual)

    def test_relu_base_quantized_behaves_equivalent_to_relu(self) -> None:
        relu = _ReLUBase(input_quant=lambda x: x + 2, input_dequant=lambda x: x - 1)

        xs = torch.arange(-10, 10)
        expected = to_list(F.relu(xs + 1))
        actual = to_list(relu(xs))

        self.assertEqual(expected, actual)

    def test_relu_base_quantized_forward_raises_error(self) -> None:
        with self.assertRaises(NotImplementedError):
            _ReLUBase().quantized_forward(torch.ones(1))


class FixedPointReLUTest(unittest.TestCase):
    def test_fixed_point_relu_forward_works_as_expected(self) -> None:
        fp_args = 8, 3
        relu = FixedPointReLU(fixed_point_factory=FixedPoint.get_factory(*fp_args))

        xs = torch.linspace(-10, 10, 100)
        fake_quantized_xs = from_fixed_point(
            to_fixed_point(to_list(xs), *fp_args), *fp_args
        )
        fake_quantized_xs = torch.tensor(fake_quantized_xs, dtype=torch.float32)
        expected = to_list(F.relu(fake_quantized_xs))
        actual = to_list(relu(xs))

        self.assertEqual(expected, actual)

    def test_fixed_point_quantized_forward_works_as_expected(self) -> None:
        relu = FixedPointReLU(
            fixed_point_factory=FixedPoint.get_factory(total_bits=8, frac_bits=3)
        )

        xs = torch.linspace(-10, 10, 20)
        expected = to_list(F.relu(xs))
        actual = to_list(relu.quantized_forward(xs))

        self.assertEqual(expected, actual)
