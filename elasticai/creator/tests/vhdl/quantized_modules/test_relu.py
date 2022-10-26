import unittest
from difflib import diff_bytes

import torch
import torch.nn.functional as F

from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    float_values_to_fixed_point,
)
from elasticai.creator.vhdl.quantized_modules.relu import FixedPointReLU, _ReLUBase


def to_list(x: torch.Tensor) -> list[float]:
    return x.detach().numpy().tolist()


def to_fixed_point(values: list[float], total_bits: int, frac_bits: int) -> list[int]:
    def to_fp(value: FixedPoint) -> int:
        return value.to_signed_int()

    return list(map(to_fp, float_values_to_fixed_point(values, total_bits, frac_bits)))


def from_fixed_point(values: list[int], total_bits: int, frac_bits: int) -> list[float]:
    def to_float(value: int) -> float:
        return float(FixedPoint.from_signed_int(value, total_bits, frac_bits))

    return list(map(to_float, values))


class ReLUBaseTest(unittest.TestCase):
    def test_relu_base_not_quantized_behaves_equivalent_to_relu(self) -> None:
        relu = _ReLUBase()

        xs = torch.linspace(-10, 10, 100)
        expected = to_list(F.relu(xs))
        actual = to_list(relu(xs))

        self.assertEquals(expected, actual)

    def test_relu_base_quantized_behaves_equivalent_to_relu(self) -> None:
        relu = _ReLUBase(input_quant=lambda x: x + 2, input_dequant=lambda x: x - 1)

        xs = torch.arange(-10, 10)
        expected = to_list(F.relu(xs + 1))
        actual = to_list(relu(xs))

        self.assertEquals(expected, actual)

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

        self.assertEquals(expected, actual)

    def test_fixed_point_quantized_forward_works_as_expected(self) -> None:
        relu = FixedPointReLU(
            fixed_point_factory=FixedPoint.get_factory(total_bits=8, frac_bits=3)
        )

        xs = torch.linspace(-10, 10, 20)
        expected = to_list(F.relu(xs))
        actual = to_list(relu.quantized_forward(xs))

        self.assertEquals(expected, actual)
