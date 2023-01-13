import unittest
from typing import Callable

import torch
import torch.nn.functional as f

from elasticai.creator.nn.hard_tanh import FixedPointHardTanh, _HardTanhBase
from elasticai.creator.tests.unit.nn.utils import (
    from_fixed_point,
    to_fixed_point,
    to_list,
)
from elasticai.creator.vhdl.number_representations import FixedPoint


class HardTanhBaseTest(unittest.TestCase):
    def test_hard_tanh_base_behaves_equal_to_hard_sigmoid(self) -> None:
        inputs = torch.linspace(-10, 10, 100)
        hard_tanh = _HardTanhBase(min_val=-2, max_val=2)
        actual = to_list(hard_tanh(inputs))
        expected = to_list(f.hardtanh(inputs, min_val=-2, max_val=2))
        self.assertEqual(expected, actual)

    def test_hard_tanh_base_input_qunat_dequant(self) -> None:
        inputs = torch.linspace(-10, 10, 100)
        hard_tanh = _HardTanhBase(
            input_quant=lambda x: x + 2, input_dequant=lambda x: x - 1
        )
        actual = to_list(hard_tanh(inputs))
        expected = to_list(f.hardtanh(inputs + 1))
        self.assertEqual(expected, actual)

    def test_hard_tanh_base_quantized_forward_raises_error(self) -> None:
        with self.assertRaises(NotImplementedError):
            _HardTanhBase().quantized_forward(torch.ones(10))


class FixedPointHardTanhTest(unittest.TestCase):
    def test_fp_hard_tanh_calculates_output_correctly(self) -> None:
        fp_args = 8, 3
        hard_tanh = FixedPointHardTanh(
            fixed_point_factory=FixedPoint.get_factory(*fp_args)
        )

        xs = torch.linspace(-10, 10, 100)
        quantized_xs = torch.tensor(
            from_fixed_point(to_fixed_point(to_list(xs), *fp_args), *fp_args),
            dtype=torch.float32,
        )
        expected = to_list(f.hardtanh(quantized_xs))
        actual = to_list(hard_tanh(xs))

        self.assertEqual(expected, actual)

    def test_fp_hard_tanh_quantized_forward(self) -> None:
        fp_args = 8, 3
        hard_tanh = FixedPointHardTanh(
            fixed_point_factory=FixedPoint.get_factory(*fp_args)
        )

        inputs = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        expected = [-8, -8, -4, 0, 4, 8, 8]

        quantized_xs = torch.tensor(
            to_fixed_point(inputs, *fp_args), dtype=torch.float32
        )
        actual = to_list(hard_tanh.quantized_forward(quantized_xs))

        self.assertEqual(expected, actual)
