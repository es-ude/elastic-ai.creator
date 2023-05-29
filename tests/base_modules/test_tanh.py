import pytest
import torch

from elasticai.creator.base_modules.float_arithmetics import FloatArithmetics
from elasticai.creator.base_modules.tanh import Tanh
from tests.tensor_test_case import assertTensorEqual


def test_single_sample() -> None:
    tanh = Tanh(arithmetics=FloatArithmetics(), num_steps=2)
    inputs = torch.tensor([-10, -3, -2, -1, 0, 1, 2, 3, 10])
    assertTensorEqual(
        expected=[-1, -1, -1, -1, -1, 1, 1, 1, 1],
        actual=tanh(inputs),
    )
