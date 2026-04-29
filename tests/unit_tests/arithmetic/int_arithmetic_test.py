from random import randint

import pytest
import torch

from elasticai.creator.arithmetic import IntArithmetic, IntParams


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("is_signed", [False, True])
def test_params(total_bits: int, is_signed: bool) -> None:
    config = IntArithmetic(IntParams(total_bits=total_bits, signed=is_signed))

    assert type(config.config) is IntParams
    assert config.config.total_bits == total_bits
    assert config.total_bits == total_bits
    assert config.config.minimum_value == config.minimum_value
    assert config.minimum_value == 0 if not is_signed else -(2 ** (total_bits - 1))
    assert config.config.maximum_value == config.maximum_value
    assert (
        config.maximum_value == 2**total_bits - 1
        if not is_signed
        else 2 ** (total_bits - 1) - 1
    )
    assert config.integer_out_of_bounds(config.config.maximum_value + 1)
    assert not config.integer_out_of_bounds(config.config.maximum_value)
    assert config.integer_out_of_bounds(config.config.minimum_value - 1)
    assert not config.integer_out_of_bounds(config.config.minimum_value)
    assert config.is_power_of_2(config.maximum_value + 1)
    assert not config.is_power_of_2(config.maximum_value)


@pytest.mark.parametrize("total_bits", [4, 8])
@pytest.mark.parametrize("is_signed", [False, True])
def test_cut_to_integer(total_bits: int, is_signed: bool) -> None:
    config = IntArithmetic(IntParams(total_bits=total_bits, signed=is_signed))

    stimuli = [
        config.config.minimum_value,
        -1.5,
        0,
        1.5,
        config.config.maximum_value,
    ]
    check = [
        config.config.minimum_value,
        -1,
        0,
        1,
        config.config.maximum_value,
    ]

    result = [config.cut_as_integer(val) for val in stimuli]
    assert result == check

    result = config.cut_as_integer(torch.asarray(stimuli)).tolist()
    assert result == check


@pytest.mark.parametrize("total_bits", [4, 8])
@pytest.mark.parametrize("is_signed", [False, True])
def test_round_to_integer(total_bits: int, is_signed: bool) -> None:
    config = IntArithmetic(IntParams(total_bits=total_bits, signed=is_signed))

    stimuli = [
        config.config.minimum_value,
        -1.4,
        0,
        1.6,
        config.config.maximum_value,
    ]
    check = [
        config.config.minimum_value,
        -1,
        0,
        2,
        config.config.maximum_value,
    ]

    result = [config.round_to_integer(val) for val in stimuli]
    assert result == check

    result = config.round_to_integer(torch.asarray(stimuli)).tolist()
    assert result == check


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("is_signed", [False, True])
def test_clamp(total_bits: int, is_signed: bool) -> None:
    sets = IntParams(total_bits=total_bits, signed=is_signed)
    config = IntArithmetic(fxp_params=sets)

    stimuli = list()
    stimuli.extend(
        sets.minimum_value - sets.total_bits + val for val in range(sets.total_bits)
    )
    stimuli.extend(sets.minimum_value + val for val in range(sets.total_bits))
    stimuli.extend(
        sets.maximum_value - sets.total_bits + val for val in range(sets.total_bits)
    )
    stimuli.extend(sets.maximum_value + val for val in range(sets.total_bits))

    check = list()
    check.extend(sets.minimum_value for _ in range(sets.total_bits))
    check.extend(sets.minimum_value + val for val in range(sets.total_bits))
    check.extend(
        sets.maximum_value - sets.total_bits + val for val in range(sets.total_bits)
    )
    check.extend(sets.maximum_value for _ in range(sets.total_bits))

    result = [config.clamp(val) for val in stimuli]
    assert result == check

    result = config.clamp(torch.asarray(stimuli)).tolist()
    assert result == check


@pytest.mark.parametrize("total_bits", [4, 8, 12])
@pytest.mark.parametrize("is_signed", [True, False])
def test_to_twos(total_bits: int, is_signed: bool) -> None:
    sets = IntArithmetic(IntParams(total_bits=total_bits, signed=is_signed))

    data = [randint(a=sets.minimum_value, b=sets.maximum_value) for _ in range(16)]
    chck = data.copy()
    for idx, val in enumerate(data):
        if val < 0:
            chck[idx] = 2**sets.total_bits + val

    rslt = [sets.to_twos(val) for val in data]
    assert rslt == chck

    rslt = sets.to_twos(torch.asarray(data)).tolist()
    assert rslt == chck
