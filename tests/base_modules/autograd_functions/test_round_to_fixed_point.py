from typing import cast

import pytest
import torch

from elasticai.creator.base_modules.autograd_functions.round_to_fixed_point import (
    RoundToFixedPoint,
)
from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from tests.tensor_test_case import assertTensorEqual


def roundToFP(inputs: list[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(inputs, list):
        inputs = torch.tensor(inputs)
    config = FixedPointConfig(total_bits=4, frac_bits=2)
    return cast(torch.Tensor, RoundToFixedPoint.apply(inputs, config))


def test_round_upper_bound() -> None:
    assertTensorEqual(
        expected=roundToFP([1.75]),
        actual=[1.75],
    )


def test_round_lower_bound() -> None:
    assertTensorEqual(
        expected=roundToFP([-2.0]),
        actual=[-2.0],
    )


def test_round_out_of_bounds() -> None:
    inputs = torch.tensor([2.0])
    with pytest.raises(ValueError):
        _ = roundToFP(inputs)


def test_round_typical_values() -> None:
    assertTensorEqual(
        expected=roundToFP([-1.3, 0.1, 1.6]),
        actual=[-1.25, 0, 1.5],
    )
