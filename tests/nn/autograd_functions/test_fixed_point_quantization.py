import torch

from elasticai.creator.nn._two_complement_fixed_point_config import FixedPointConfig
from elasticai.creator.nn.autograd_functions.fixed_point_quantization import (
    FixedPointDequantFunction,
    FixedPointQuantFunction,
)
from tests.tensor_test_case import TensorTestCase


class FixedPointQuantFunctionTest(TensorTestCase):
    def setUp(self) -> None:
        self.fp_factory = FixedPointConfig(total_bits=4, frac_bits=2)
        self.quant = lambda x: FixedPointQuantFunction.apply(x, self.fp_factory)

    def test_quantize_upper_bound(self) -> None:
        x = torch.tensor([1.75])
        actual = self.quant(x)
        target = [7.0]
        self.assertTensorEqual(target, actual)

    def test_quantize_lower_bound(self) -> None:
        x = torch.tensor([-2.0])
        actual = self.quant(x)
        target = [-8.0]
        self.assertTensorEqual(target, actual)

    def test_quantize_out_of_bounds(self) -> None:
        x = torch.tensor([2.0])
        with self.assertRaises(ValueError):
            _ = self.quant(x)

    def test_quantize_typical_values(self) -> None:
        x = torch.tensor([-1.3, 0.1, 1.6])
        actual = self.quant(x)
        target = [-5, 0, 6]
        self.assertTensorEqual(target, actual)


class FixedPointDequantFunctionTest(TensorTestCase):
    def setUp(self) -> None:
        self.fp_factory: FixedPointConfig = FixedPointConfig(total_bits=4, frac_bits=2)
        self.dequant = lambda x: FixedPointDequantFunction.apply(x, self.fp_factory)

    def test_dequantize_upper_bound(self) -> None:
        x = torch.tensor([7.0])
        actual = self.dequant(x)
        target = [1.75]
        self.assertTensorEqual(target, actual)

    def test_dequantize_lower_bound(self) -> None:
        x = torch.tensor([-8.0])
        actual = self.dequant(x)
        target = [-2.0]
        self.assertTensorEqual(target, actual)

    def test_dequantize_out_of_bounds(self) -> None:
        def dequant(x):
            return FixedPointDequantFunction.apply(
                x, FixedPointConfig(total_bits=4, frac_bits=2)
            )

        x = torch.tensor([9])
        with self.assertRaises(ValueError):
            _ = dequant(x)

    def test_dequantize_typical_values(self) -> None:
        x = torch.tensor([-5, 0, 6])
        actual = self.dequant(x)
        target = [-1.25, 0, 1.5]
        self.assertTensorEqual(target, actual)
