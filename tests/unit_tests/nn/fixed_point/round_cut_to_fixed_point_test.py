from typing import cast

import pytest
import torch

from elasticai.creator.nn.fixed_point.round_to_fixed_point import (
    CutToFixedPoint,
    RoundToFixedPoint,
)
from elasticai.creator.nn.fixed_point.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from tests.tensor_test_case import assertTensorEqual


def round_to_fxp(inputs: list[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(inputs, list):
        inputs = torch.tensor(inputs)
    return cast(
        torch.Tensor,
        RoundToFixedPoint.apply(inputs, FixedPointConfig(total_bits=4, frac_bits=2)),
    )


def cut_to_fxp(inputs: list[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(inputs, list):
        inputs = torch.tensor(inputs)
    return cast(
        torch.Tensor,
        CutToFixedPoint.apply(inputs, FixedPointConfig(total_bits=4, frac_bits=2)),
    )


def test_round_upper_bound() -> None:
    assertTensorEqual(
        expected=round_to_fxp([1.75]),
        actual=[1.75],
    )


def test_round_lower_bound() -> None:
    assertTensorEqual(
        expected=round_to_fxp([-2.0]),
        actual=[-2.0],
    )


def test_round_out_of_bounds() -> None:
    inputs = torch.tensor([2.0])
    with pytest.raises(ValueError):
        _ = round_to_fxp(inputs)


def test_round_typical_values() -> None:
    assertTensorEqual(
        expected=round_to_fxp([-1.3, 0.1, 1.6]),
        actual=[-1.25, 0, 1.5],
    )


def test_cut_upper_bound() -> None:
    assertTensorEqual(
        expected=cut_to_fxp([1.75]),
        actual=[1.75],
    )


def test_cut_lower_bound() -> None:
    assertTensorEqual(
        expected=cut_to_fxp([-2.0]),
        actual=[-2.0],
    )


def test_cut_out_of_bounds() -> None:
    inputs = torch.tensor([2.0])
    with pytest.raises(ValueError):
        _ = cut_to_fxp(inputs)


def test_cut_typical_values() -> None:
    assertTensorEqual(
        expected=cut_to_fxp([-1.3, 0.1, 1.6]),
        actual=[-1.25, 0, 1.5],
    )
