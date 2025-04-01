from typing import cast

import pytest
import torch

from elasticai.creator.nn.float.round_to_float import RoundToFloat
from tests.tensor_test_case import assertTensorEqual


def roundToFloat(
    inputs: float | list[float] | torch.Tensor, mantissa_bits: int, exponent_bits: int
) -> torch.Tensor:
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs)
    return cast(torch.Tensor, RoundToFloat.apply(inputs, mantissa_bits, exponent_bits))


@pytest.mark.parametrize(
    "decimal,target_float,mantissa_bits,exponent_bits",
    [
        (0.0, 0.001953125, 6, 3),
        (9.5694284, 9.625, 6, 3),
        (-9.5694284, -9.625, 6, 3),
        (0.0, 0.125, 3, 1),
        (1.525, 1.5, 3, 1),
    ],
)
def test_round_single_number_to_correct_float(
    decimal: float, target_float: float, mantissa_bits: int, exponent_bits: int
) -> None:
    actual_float = roundToFloat(decimal, mantissa_bits, exponent_bits).item()
    assert actual_float == target_float


def test_uses_different_scale_for_each_value_in_tensor() -> None:
    values = [-14.9, -5.4, 0.05, 8.123456789]
    target_floats = [-14.875, -5.375, 0.0498046875, 8.125]
    actual_floats = roundToFloat(values, 6, 3)
    assertTensorEqual(target_floats, actual_floats)


@pytest.mark.parametrize("decimal", [42.424242, -25.12345678])
def test_raises_error_if_value_out_of_bounds(decimal: float) -> None:
    with pytest.raises(ValueError):
        _ = roundToFloat(decimal, 6, 3)
