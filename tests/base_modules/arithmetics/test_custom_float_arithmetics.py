import torch

from elasticai.creator.base_modules.arithmetics.custom_float_arithmetics import (
    CustomFloatArithmetics,
)
from tests.tensor_test_case import assertTensorEqual


def test_clamp() -> None:
    arithmetics = CustomFloatArithmetics(mantissa_bits=3, exponent_bits=1)
    a = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    assertTensorEqual(
        expected=[-1.875, -1.875, -1.0, 0.0, 1.0, 1.875, 1.875],
        actual=arithmetics.clamp(a),
    )


def test_round() -> None:
    arithmetics = CustomFloatArithmetics(mantissa_bits=3, exponent_bits=1)
    a = torch.tensor([-1.69, -0.2, 0.2, 1.69])
    assertTensorEqual(
        expected=[-1.75, -0.203125, 0.203125, 1.75],
        actual=arithmetics.round(a),
    )


def test_quantize() -> None:
    arithmetics = CustomFloatArithmetics(mantissa_bits=3, exponent_bits=1)
    a = torch.tensor([-3.0, -2.0, -1.69, -0.2, 0.2, 1.69, 2.0, 3.0])
    assertTensorEqual(
        expected=[-1.875, -1.875, -1.75, -0.203125, 0.203125, 1.75, 1.875, 1.875],
        actual=arithmetics.quantize(a),
    )


def test_add() -> None:
    arithmetics = CustomFloatArithmetics(mantissa_bits=3, exponent_bits=1)
    a = torch.tensor([-0.6875, 0.1250, 0.6875])
    b = torch.tensor([-0.3125, 0.2031, 0.3125])
    assertTensorEqual(
        expected=[-1.0, 0.3125, 1.0],
        actual=arithmetics.add(a, b),
    )


def test_sum() -> None:
    arithmetics = CustomFloatArithmetics(mantissa_bits=3, exponent_bits=1)
    a = torch.tensor([-1.0, 0.3125, 1.0])
    assertTensorEqual(
        expected=0.3125,
        actual=arithmetics.sum(a),
    )


def test_mul() -> None:
    arithmetics = CustomFloatArithmetics(mantissa_bits=3, exponent_bits=1)
    a = torch.tensor([-0.6875, 0.1250, 0.6875])
    b = torch.tensor([-0.3125, 0.2031, 0.3125])
    assertTensorEqual(
        expected=[0.21875, 0.125, 0.21875],
        actual=arithmetics.mul(a, b),
    )


def test_matmul() -> None:
    arithmetics = CustomFloatArithmetics(mantissa_bits=3, exponent_bits=1)
    a = torch.tensor([[-2.0, -1.75, -1.5], [-0.25, 0.0, 0.25], [1.25, 1.5, 1.75]])
    b = torch.tensor([[-0.25], [0.5], [0.25]])
    assertTensorEqual(
        expected=[[-0.75], [0.125], [0.875]],
        actual=arithmetics.matmul(a, b),
    )


def test_conv1d() -> None:
    arithmetics = CustomFloatArithmetics(mantissa_bits=3, exponent_bits=1)
    a = torch.tensor([[-1.75, -1.5, -1, -0.25, 1, 2.5, 3.75]])
    assertTensorEqual(
        expected=[[-1.875, -1.5, -0.25, 1.75, 1.875, 1.875]],
        actual=arithmetics.conv1d(
            inputs=a,
            weights=torch.ones(1, 1, 2),
            bias=torch.ones(1),
            stride=1,
            padding="valid",
            dilation=1,
            groups=1,
        ),
    )
