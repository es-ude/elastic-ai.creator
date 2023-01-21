import torch
from torch.nn.parameter import Parameter

from elasticai.creator.nn.arithmetics import FloatArithmetics
from elasticai.creator.nn.linear import FixedPointLinear, _LinearBase
from elasticai.creator.nn.quantization import QuantType
from elasticai.creator.tests.unit.tensor_test_case import TensorTestCase
from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointFactory


def tensor(data: list) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float32)


def linear_base_with_fixed_params(
    in_features: int,
    out_features: int,
    bias: bool,
    input_quant: QuantType = lambda x: x,
    param_quant: QuantType = lambda x: x,
) -> _LinearBase:
    arithmetics = FloatArithmetics()
    linear = _LinearBase(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        arithmetics=arithmetics,
        input_quant=input_quant,
        param_quant=param_quant,
    )
    linear.weight = Parameter(torch.ones_like(linear.weight))
    if bias:
        linear.bias = Parameter(torch.ones_like(linear.bias))
    return linear


def fp_linear_with_fixed_params(
    in_features: int,
    out_features: int,
    bias: bool,
    fixed_point_factory: FixedPointFactory,
) -> FixedPointLinear:
    linear = FixedPointLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        fixed_point_factory=fixed_point_factory,
    )
    linear.weight = Parameter(torch.ones_like(linear.weight))
    if bias:
        linear.bias = Parameter(torch.ones_like(linear.bias))
    return linear


class LinearBaseTest(TensorTestCase):
    def test_with_bias(self) -> None:
        linear = linear_base_with_fixed_params(in_features=3, out_features=1, bias=True)

        actual = linear(tensor([1, 2, 3]))
        expected = [7.0]

        self.assertTensorEqual(expected, actual)

    def test_without_bias(self) -> None:
        linear = linear_base_with_fixed_params(
            in_features=3, out_features=1, bias=False
        )

        actual = linear(tensor([1, 2, 3]))
        expected = [6.0]

        self.assertTensorEqual(expected, actual)

    def test_input_quant(self) -> None:
        linear = linear_base_with_fixed_params(
            in_features=3, out_features=1, bias=True, input_quant=lambda x: x + 3
        )

        actual = linear(tensor([1, 2, 3]))
        expected = [16.0]

        self.assertTensorEqual(expected, actual)

    def test_param_quant(self) -> None:
        linear = linear_base_with_fixed_params(
            in_features=3, out_features=1, bias=True, param_quant=lambda x: x + 3
        )

        actual = linear(tensor([1, 2, 3]))
        expected = [28.0]

        self.assertTensorEqual(expected, actual)

    def test_input_param_quant(self) -> None:
        linear = linear_base_with_fixed_params(
            in_features=3,
            out_features=1,
            bias=True,
            input_quant=lambda x: x + 1,
            param_quant=lambda x: x + 3,
        )

        actual = linear(tensor([1, 2, 3]))
        expected = [40.0]

        self.assertTensorEqual(expected, actual)


class FixedPointLinearTest(TensorTestCase):
    def test_with_inputs_in_bounds(self) -> None:
        fp_factory = FixedPoint.get_factory(total_bits=8, frac_bits=2)
        linear = fp_linear_with_fixed_params(
            in_features=3,
            out_features=1,
            bias=True,
            fixed_point_factory=fp_factory,
        )

        actual = linear(tensor([-3.25, 1.5, 0.25]))
        expected = [-0.5]

        self.assertTensorEqual(expected, actual)

    def test_with_inputs_out_of_bounds(self) -> None:
        fp_factory = FixedPoint.get_factory(total_bits=2, frac_bits=0)
        linear = fp_linear_with_fixed_params(
            in_features=3,
            out_features=1,
            bias=True,
            fixed_point_factory=fp_factory,
        )

        actual = linear(tensor([1, 2, 3]))
        expected = [1.0]

        self.assertTensorEqual(expected, actual)
