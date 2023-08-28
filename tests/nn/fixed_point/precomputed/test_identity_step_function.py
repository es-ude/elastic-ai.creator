from collections.abc import Iterable
from typing import cast

import pytest
import torch

from elasticai.creator.nn.fixed_point.precomputed.identity_step_function import (
    IdentityStepFunction,
)
from tests.tensor_test_case import assertTensorEqual


def generate_step_lut(min: float, max: float, steps: int) -> torch.Tensor:
    return torch.linspace(min, max, steps)


@pytest.mark.parametrize(
    "minimum,maximum,steps,inputs,outputs",
    [
        (-3, 3, 2, range(-4, 5), [-3, -3, 3, 3, 3, 3, 3, 3, 3]),
        (-3, 3, 3, range(-4, 5), [-3, -3, 0, 0, 0, 3, 3, 3, 3]),
        (-3, 3, 3, range(5, 10), [3, 3, 3, 3, 3]),
        (-5, -2, 3, range(-6, 0), [-5, -5, -3.5, -2, -2, -2]),
        (2, 5, 3, range(1, 7), [2, 2, 3.5, 5, 5, 5]),
    ],
)
def test_inputs_correctly_mapped_to_step_function_inputs(
    minimum: float,
    maximum: float,
    steps: int,
    inputs: Iterable[float],
    outputs: Iterable[float],
) -> None:
    step_lut = generate_step_lut(minimum, maximum, steps)
    actual_outputs = cast(
        torch.Tensor, IdentityStepFunction.apply(torch.tensor(inputs), step_lut)
    )
    assertTensorEqual(list(outputs), actual_outputs)


@pytest.mark.parametrize("steps", [1, 0])
def test_raises_error_when_steps_less_than_or_equal_one(steps: int) -> None:
    inputs = torch.tensor([1, 2])
    step_lut = generate_step_lut(-1, 1, steps)
    with pytest.raises(ValueError):
        IdentityStepFunction.apply(inputs, step_lut)
