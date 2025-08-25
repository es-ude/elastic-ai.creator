from typing import cast

import pytest
import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.nn.fixed_point.fxp_round_cut import (
    CutToFixedPoint,
    RoundToFixedPoint,
)
from tests.tensor_test_case import assertTensorEqual


def round_to_fxp(inputs: list[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(inputs, list):
        inputs = torch.tensor(inputs)
    return cast(
        torch.Tensor,
        RoundToFixedPoint.apply(
            inputs, FxpArithmetic(FxpParams(total_bits=4, frac_bits=2, signed=True))
        ),
    )


def cut_to_fxp(inputs: list[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(inputs, list):
        inputs = torch.tensor(inputs)
    return cast(
        torch.Tensor,
        CutToFixedPoint.apply(
            inputs, FxpArithmetic(FxpParams(total_bits=4, frac_bits=2, signed=True))
        ),
    )


@pytest.mark.parametrize(
    "val_in, val_out",
    [(1.75, 1.75), (-2.0, -2.0), (-1.3, -1.25), (0.1, 0.0), (1.6, 1.5)],
)
def test_round_integer(val_in: float, val_out: float) -> None:
    assertTensorEqual(
        expected=round_to_fxp([val_in]),
        actual=[val_out],
    )


@pytest.mark.parametrize("val_in, val_out", [(-2.5, -2.0), (2.0, 1.75)])
def test_round_out_of_bounds(val_in: float, val_out: float) -> None:
    result = torch.tensor([val_in])
    with pytest.raises(ValueError):
        _ = round_to_fxp(result)


@pytest.mark.parametrize(
    "val_in, val_out",
    [(1.75, 1.75), (-2.0, -2.0), (-1.3, -1.25), (0.1, 0.0), (1.6, 1.5)],
)
def test_cut_integer(val_in: float, val_out: float) -> None:
    assertTensorEqual(
        expected=cut_to_fxp([val_in]),
        actual=[val_out],
    )


@pytest.mark.parametrize("val_in, val_out", [(-2.5, -2.0), (2.0, 1.75)])
def test_cut_out_of_bounds(val_in: float, val_out: float) -> None:
    inputs = torch.tensor([val_in])
    with pytest.raises(ValueError):
        _ = cut_to_fxp(inputs)
