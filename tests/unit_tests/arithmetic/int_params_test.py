import pytest
import torch

from elasticai.creator.arithmetic.int_params import IntParams


@pytest.mark.parametrize(
    "total_bits, signed, expected",
    [
        (2, True, -2),
        (2, False, 0),
        (3, True, -4),
        (3, False, 0),
        (5, True, -16),
        (5, False, 0),
    ],
)
def test_minimum(total_bits: int, signed: bool, expected: int) -> None:
    result = IntParams(total_bits=total_bits, signed=signed).minimum_value
    assert result == expected


@pytest.mark.parametrize(
    "total_bits, signed, expected",
    [
        (2, True, 1),
        (2, False, 3),
        (3, True, 3),
        (3, False, 7),
        (5, True, 15),
        (5, False, 31),
    ],
)
def test_maximum_as_integer(total_bits: int, signed: bool, expected: int) -> None:
    result = IntParams(total_bits=total_bits, signed=signed).maximum_value
    assert result == expected


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("is_signed", [False, True])
def test_integer_out_of_bounds(total_bits: int, is_signed: bool) -> None:
    sets = IntParams(total_bits=total_bits, signed=is_signed)

    stimuli = list()
    stimuli.extend(
        sets.minimum_value - sets.total_bits + val for val in range(sets.total_bits)
    )
    stimuli.extend(sets.minimum_value + val for val in range(sets.total_bits))
    stimuli.extend(
        sets.maximum_value - sets.total_bits + val for val in range(sets.total_bits)
    )
    stimuli.extend(sets.maximum_value + 1 + val for val in range(sets.total_bits))

    check = list()
    check.extend(True for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(True for _ in range(sets.total_bits))

    rslt = [sets.integer_out_of_bounds(val) for val in stimuli]
    assert rslt == check

    rslt = sets.integer_out_of_bounds(torch.asarray(stimuli)).tolist()
    assert rslt == check


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("is_signed", [False, True])
def test_integer_overflow(total_bits: int, is_signed: bool) -> None:
    sets = IntParams(total_bits=total_bits, signed=is_signed)

    stimuli = list()
    stimuli.extend(
        sets.minimum_value - sets.total_bits + val for val in range(sets.total_bits)
    )
    stimuli.extend(sets.minimum_value + val for val in range(sets.total_bits))
    stimuli.extend(
        sets.maximum_value - sets.total_bits + val for val in range(sets.total_bits)
    )
    stimuli.extend(sets.maximum_value + 1 + val for val in range(sets.total_bits))

    check = list()
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(True for _ in range(sets.total_bits))

    rslt = [sets.integer_out_overflow(val) for val in stimuli]
    assert rslt == check

    rslt = sets.integer_out_overflow(torch.asarray(stimuli)).tolist()
    assert rslt == check


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("is_signed", [False, True])
def test_integer_underflow(total_bits: int, is_signed: bool) -> None:
    sets = IntParams(total_bits=total_bits, signed=is_signed)

    stimuli = list()
    stimuli.extend(
        sets.minimum_value - sets.total_bits + val for val in range(sets.total_bits)
    )
    stimuli.extend(sets.minimum_value + val for val in range(sets.total_bits))
    stimuli.extend(
        sets.maximum_value - sets.total_bits + val for val in range(sets.total_bits)
    )
    stimuli.extend(sets.maximum_value + 1 + val for val in range(sets.total_bits))

    check = list()
    check.extend(True for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))

    rslt = [sets.integer_out_underflow(val) for val in stimuli]
    assert rslt == check

    rslt = sets.integer_out_underflow(torch.asarray(stimuli)).tolist()
    assert rslt == check


@pytest.mark.parametrize("total_bits", [8])
@pytest.mark.parametrize("is_signed", [True, False])
def test_is_power_2(total_bits: int, is_signed: bool) -> None:
    sets = IntParams(total_bits=total_bits, signed=is_signed)

    if not is_signed:
        data_in = [0, 1, 2, 3, 4, 6, 8, 12, 16, 32, 63, 64]
        check = [
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
        ]
    else:
        data_in = [-128, -64, -16, -12, -8, 0, 8, 12, 16, 32, 63, 64]
        check = [
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
        ]

    results = [sets.is_power_of_2(val) for val in data_in]
    assert results == check

    results = sets.is_power_of_2(torch.asarray(data_in)).tolist()
    assert results == check
