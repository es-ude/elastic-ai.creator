import pytest
import torch

from elasticai.creator.nn.fixed_point.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from tests.tensor_test_case import assertTensorEqual


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_cut_float_to_integer(total_bits: int, frac_bits: int) -> None:
    config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
    chck = [config.minimum_as_integer, -1, 0, +1, config.maximum_as_integer]
    stimuli = [
        config.minimum_as_rational,
        -config.minimum_step_as_rational,
        0,
        config.minimum_step_as_rational,
        config.maximum_as_rational,
    ]
    rslt = [config.cut_as_integer(val) for val in stimuli]
    assert rslt == chck


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_cut_T_to_integer(total_bits: int, frac_bits: int) -> None:
    config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)

    chck = torch.Tensor(
        [config.minimum_as_integer, -1, 0, +1, config.maximum_as_integer]
    )
    stimuli = torch.Tensor(
        [
            config.minimum_as_rational,
            -config.minimum_step_as_rational,
            0,
            config.minimum_step_as_rational,
            config.maximum_as_rational,
        ]
    )
    rslt = config.cut_as_integer(stimuli)
    assertTensorEqual(rslt, chck)


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_cut_x_to_integer(total_bits: int, frac_bits: int) -> None:
    config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)

    stimuli_tensor = torch.Tensor(
        [
            config.minimum_as_rational,
            -config.minimum_step_as_rational,
            0,
            config.minimum_step_as_rational,
            config.maximum_as_rational,
        ]
    )
    stimuli_float = [
        config.minimum_as_rational,
        -config.minimum_step_as_rational,
        0,
        config.minimum_step_as_rational,
        config.maximum_as_rational,
    ]
    rslt_tensor = config.cut_as_integer(stimuli_tensor).tolist()
    rslt_float = [config.cut_as_integer(val) for val in stimuli_float]
    assert rslt_tensor == rslt_float


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_round_float_to_integer(total_bits: int, frac_bits: int) -> None:
    config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
    chck = [config.minimum_as_integer, -1, 0, +1, config.maximum_as_integer]
    stimuli = [
        config.minimum_as_rational,
        -config.minimum_step_as_rational,
        0,
        config.minimum_step_as_rational,
        config.maximum_as_rational,
    ]
    rslt = [config.round_to_integer(val) for val in stimuli]
    assert rslt == chck


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_round_T_to_integer(total_bits: int, frac_bits: int) -> None:
    config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)

    chck = torch.Tensor(
        [config.minimum_as_integer, -1, 0, +1, config.maximum_as_integer]
    )
    stimuli = torch.Tensor(
        [
            config.minimum_as_rational,
            -config.minimum_step_as_rational,
            0,
            config.minimum_step_as_rational,
            config.maximum_as_rational,
        ]
    )
    rslt = config.round_to_integer(stimuli)
    assertTensorEqual(rslt, chck)


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_round_x_to_integer(total_bits: int, frac_bits: int) -> None:
    config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)

    stimuli_tensor = torch.Tensor(
        [
            config.minimum_as_rational,
            -config.minimum_step_as_rational,
            0,
            config.minimum_step_as_rational,
            config.maximum_as_rational,
        ]
    )
    stimuli_float = [
        config.minimum_as_rational,
        -config.minimum_step_as_rational,
        0,
        config.minimum_step_as_rational,
        config.maximum_as_rational,
    ]
    rslt_tensor = config.round_to_integer(stimuli_tensor).tolist()
    rslt_float = [config.round_to_integer(val) for val in stimuli_float]
    assert rslt_tensor == rslt_float
