from collections.abc import Iterable
from typing import cast

import pytest
import torch

from elasticai.creator.base_modules.autograd_functions.step_function_inputs import (
    StepFunctionInputs,
)
from tests.tensor_test_case import assertTensorEqual


@pytest.mark.parametrize(
    "minimum,maximum,steps,inputs,outputs",
    [
        (-3, 3, 2, range(-4, 5), [-3, -3, 3, 3, 3, 3, 3, 3, 3]),
        (-3, 3, 3, range(-4, 5), [-3, -3, 0, 0, 0, 3, 3, 3, 3]),
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
    actual_outputs = cast(
        torch.Tensor,
        StepFunctionInputs.apply(
            torch.tensor(inputs, dtype=torch.float32), minimum, maximum, steps
        ),
    )
    assertTensorEqual(list(outputs), actual_outputs)


@pytest.mark.parametrize("steps", [1, 0])
def test_raises_error_when_steps_less_than_or_equal_one(steps: int) -> None:
    inputs = torch.tensor([1, 2], dtype=torch.float32)
    with pytest.raises(ValueError):
        StepFunctionInputs.apply(inputs, -1, 1, steps)
