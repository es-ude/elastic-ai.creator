from random import randint

import pytest
import torch

from elasticai.creator.arithmetic._int_arithmetic import int_arithmetic


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("is_signed", [False, True])
def test_params(total_bits: int, is_signed: bool) -> None:
    config = int_arithmetic(total_bits=total_bits, signed=is_signed)

    assert config.minimum_as_integer == 0 if not is_signed else -(2 ** (total_bits - 1))
    assert (
        config.maximum_as_integer == 2**total_bits - 1
        if not is_signed
        else 2 ** (total_bits - 1) - 1
    )
    assert config.integer_out_of_bounds(config.maximum_as_integer + 1)
    assert not config.integer_out_of_bounds(config.config.maximum_as_integer)

    assert config.integer_out_of_bounds(config.config.minimum_as_integer - 1)
    assert not config.integer_out_of_bounds(config.config.minimum_as_integer)

    assert config.is_power_of_2(config.maximum_as_integer + 1)
    assert not config.is_power_of_2(config.maximum_as_integer)


@pytest.mark.parametrize("total_bits", [4, 8])
@pytest.mark.parametrize("is_signed", [False, True])
def test_cut_to_integer(total_bits: int, is_signed: bool) -> None:
    config = int_arithmetic(total_bits=total_bits, signed=is_signed)

    stimuli = [
        config.config.minimum_as_integer,
        -1.5,
        0,
        1.5,
        config.config.maximum_as_integer,
    ]
    check = [
        config.config.minimum_as_integer,
        -1,
        0,
        1,
        config.config.maximum_as_integer,
    ]

    result = [config.cut_as_integer(val) for val in stimuli]
    assert result == check

    result = config.cut_as_integer(torch.asarray(stimuli)).tolist()
    assert result == check


@pytest.mark.parametrize("total_bits", [4, 8])
@pytest.mark.parametrize("is_signed", [False, True])
def test_round_to_integer(total_bits: int, is_signed: bool) -> None:
    config = int_arithmetic(total_bits=total_bits, signed=is_signed)

    stimuli = [
        config.config.minimum_as_integer,
        -1.4,
        0,
        1.6,
        config.config.maximum_as_integer,
    ]
    check = [
        config.config.minimum_as_integer,
        -1,
        0,
        2,
        config.config.maximum_as_integer,
    ]

    result = [config.round_to_integer(val) for val in stimuli]
    assert result == check

    result = config.round_to_integer(torch.asarray(stimuli)).tolist()
    assert result == check


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("is_signed", [False, True])
def test_clamp(total_bits: int, is_signed: bool) -> None:
    config = int_arithmetic(total_bits=total_bits, signed=is_signed)

    stimuli = list()
    stimuli.extend(
        config.minimum_as_integer - config.total_bits + val
        for val in range(config.total_bits)
    )
    stimuli.extend(config.minimum_as_integer + val for val in range(config.total_bits))
    stimuli.extend(
        config.maximum_as_integer - config.total_bits + val
        for val in range(config.total_bits)
    )
    stimuli.extend(config.maximum_as_integer + val for val in range(config.total_bits))

    check = list()
    check.extend(config.minimum_as_integer for _ in range(config.total_bits))
    check.extend(config.minimum_as_integer + val for val in range(config.total_bits))
    check.extend(
        config.maximum_as_integer - config.total_bits + val
        for val in range(config.total_bits)
    )
    check.extend(config.maximum_as_integer for _ in range(config.total_bits))

    result = [config.clamp(val) for val in stimuli]
    assert result == check

    result = config.clamp(torch.asarray(stimuli)).tolist()
    assert result == check


@pytest.mark.parametrize("total_bits", [4, 8, 12])
@pytest.mark.parametrize("is_signed", [True, False])
def test_to_twos(total_bits: int, is_signed: bool) -> None:
    config = int_arithmetic(total_bits=total_bits, signed=is_signed)

    data = [
        randint(a=config.minimum_as_integer, b=config.maximum_as_integer)
        for _ in range(16)
    ]
    chck = data.copy()
    for idx, val in enumerate(data):
        if val < 0:
            chck[idx] = 2**config.total_bits + val

    rslt = [config.to_twos(val) for val in data]
    assert rslt == chck

    rslt = config.to_twos(torch.asarray(data)).tolist()
    assert rslt == chck
