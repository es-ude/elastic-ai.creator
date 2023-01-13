import unittest

import torch
import torch.nn.functional as F

from elasticai.creator.nn.hard_sigmoid import FixedPointHardSigmoid, _HardSigmoidBase
from elasticai.creator.tests.unit.nn.utils import (
    from_fixed_point,
    to_fixed_point,
    to_list,
)
from elasticai.creator.vhdl.number_representations import FixedPoint


class HardSigmoidBaseTest(unittest.TestCase):
    def test_hard_sigmoid_base_behaves_equal_to_hard_sigmoid(self) -> None:
        hard_sigmoid = _HardSigmoidBase()

        xs = torch.linspace(-6, 6, 100)
        expected = to_list(F.hardsigmoid(xs))
        actual = to_list(hard_sigmoid(xs))

        self.assertEqual(expected, actual)

    def test_hard_sigmoid_base_input_qunat_dequant(self) -> None:
        hard_sigmoid = _HardSigmoidBase(
            input_quant=lambda x: x * 4,
            input_dequant=lambda x: x / 2,
        )

        xs = torch.tensor([-1, 0, 1], dtype=torch.float32)
        expected = to_list(F.hardsigmoid(xs * 2))
        actual = to_list(hard_sigmoid(xs))

        self.assertEqual(expected, actual)

    def test_hard_sigmoid_base_quantized_forward_raises_error(self) -> None:
        with self.assertRaises(NotImplementedError):
            _HardSigmoidBase().quantized_forward(torch.ones(10))


class FixedPointHardSigmoidTest(unittest.TestCase):
    def test_fixed_point_hard_sigmoid_calculates_output_correctly(self) -> None:
        fp_args = 8, 3
        hard_sigmoid = FixedPointHardSigmoid(
            fixed_point_factory=FixedPoint.get_factory(*fp_args)
        )

        xs = torch.linspace(-6, 6, 100)
        quantized_xs = torch.tensor(
            from_fixed_point(to_fixed_point(to_list(xs), *fp_args), *fp_args),
            dtype=torch.float32,
        )
        expected = to_list(F.hardsigmoid(quantized_xs))
        actual = to_list(hard_sigmoid(xs))

        self.assertEqual(expected, actual)

    def test_fixed_point_hard_sigmoid_quantized_forward(self) -> None:
        fp_args = 8, 3
        hard_sigmoid = FixedPointHardSigmoid(
            fixed_point_factory=FixedPoint.get_factory(*fp_args)
        )

        xs = [-3, -1.5, 0, 1.5, 3]
        expected = [0, 3, 4, 5, 8]

        quantized_xs = torch.tensor(to_fixed_point(xs, *fp_args), dtype=torch.float32)
        actual = to_list(hard_sigmoid.quantized_forward(quantized_xs))

        self.assertEqual(expected, actual)
