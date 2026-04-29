import pytest

from elasticai.creator.arithmetic.fxp_converter import (
    FxpParams,
)


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, expected",
    [
        (2, 1, True, -2),
        (2, 1, False, 0),
        (3, 1, True, -4),
        (3, 1, False, 0),
        (5, 1, True, -16),
        (5, 1, False, 0),
    ],
)
def test_fxp_minimum_as_integer(
    total_bits: int, frac_bits: int, signed: bool, expected: int
) -> None:
    result = FxpParams(
        total_bits=total_bits, frac_bits=frac_bits, signed=signed
    ).minimum_as_integer
    assert result == expected


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, expected",
    [
        (2, 1, True, 1),
        (2, 1, False, 3),
        (3, 1, True, 3),
        (3, 1, False, 7),
        (5, 1, True, 15),
        (5, 1, False, 31),
    ],
)
def test_fxp_maximum_as_integer(
    total_bits: int, frac_bits: int, signed: bool, expected: int
) -> None:
    result = FxpParams(
        total_bits=total_bits, frac_bits=frac_bits, signed=signed
    ).maximum_as_integer
    assert result == expected


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, expected",
    [
        (2, 0, True, -2.0),
        (2, 1, True, -1.0),
        (2, 1, False, 0.0),
        (3, 1, True, -2.0),
        (3, 1, False, 0.0),
        (5, 1, True, -8.0),
        (5, 1, False, 0.0),
    ],
)
def test_fxp_minimum_as_rational(
    total_bits: int, frac_bits: int, signed: bool, expected: int
) -> None:
    result = FxpParams(
        total_bits=total_bits, frac_bits=frac_bits, signed=signed
    ).minimum_as_rational
    assert result == expected


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, expected",
    [
        (2, 1, True, 0.5),
        (2, 2, True, 0.25),
        (2, 2, False, 0.25),
        (3, 1, True, 0.5),
        (3, 2, True, 0.25),
        (3, 2, False, 0.25),
        (5, 3, True, 0.125),
        (5, 4, False, 0.0625),
    ],
)
def test_fxp_minimum_step_as_rational(
    total_bits: int, frac_bits: int, signed: bool, expected: int
) -> None:
    result = FxpParams(
        total_bits=total_bits, frac_bits=frac_bits, signed=signed
    ).minimum_step_as_rational
    assert result == expected


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, expected",
    [
        (2, 0, True, 1.0),
        (2, 1, True, 0.5),
        (2, 1, False, 1.5),
        (3, 2, True, 0.75),
        (3, 2, False, 1.75),
        (5, 2, True, 3.75),
        (5, 3, False, 3.875),
    ],
)
def test_fxp_maximum_as_rational(
    total_bits: int, frac_bits: int, signed: bool, expected: int
) -> None:
    result = FxpParams(
        total_bits=total_bits, frac_bits=frac_bits, signed=signed
    ).maximum_as_rational
    assert result == expected


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("frac_bits", [0, 1, 2, 3])
@pytest.mark.parametrize("is_signed", [False, True])
def test_integer_out_of_bounds(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    sets = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)

    stimuli = list()
    stimuli.extend(sets.minimum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.minimum_as_integer + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer + 1 + val for val in range(sets.total_bits))

    check = list()
    check.extend(True for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(True for _ in range(sets.total_bits))

    rslt = [sets.integer_out_of_bounds(val) for val in stimuli]
    assert rslt == check


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("frac_bits", [0, 1, 2, 3])
@pytest.mark.parametrize("is_signed", [False, True])
def test_integer_overflow(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    sets = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)

    stimuli = list()
    stimuli.extend(sets.minimum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.minimum_as_integer + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer + 1 + val for val in range(sets.total_bits))

    check = list()
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(True for _ in range(sets.total_bits))

    rslt = [sets.integer_out_overflow(val) for val in stimuli]
    assert rslt == check


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("frac_bits", [0, 1, 2, 3])
@pytest.mark.parametrize("is_signed", [False, True])
def test_integer_underflow(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    sets = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)

    stimuli = list()
    stimuli.extend(sets.minimum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.minimum_as_integer + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer + 1 + val for val in range(sets.total_bits))

    check = list()
    check.extend(True for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))

    rslt = [sets.integer_out_underflow(val) for val in stimuli]
    assert rslt == check


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("frac_bits", [0, 1, 2, 3])
@pytest.mark.parametrize("is_signed", [False, True])
def test_rational_out_of_bounds(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    sets = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)

    stimuli = list()
    stimuli.extend(sets.minimum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.minimum_as_integer + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer + 1 + val for val in range(sets.total_bits))

    check = list()
    check.extend(True for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(True for _ in range(sets.total_bits))

    rslt = [sets.rational_out_of_bounds(val * sets.minimum_step_as_rational) for val in stimuli]
    assert rslt == check


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("frac_bits", [0, 1, 2, 3])
@pytest.mark.parametrize("is_signed", [False, True])
def test_rational_overflow(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    sets = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)

    stimuli = list()
    stimuli.extend(sets.minimum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.minimum_as_integer + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer + 1 + val for val in range(sets.total_bits))

    check = list()
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(True for _ in range(sets.total_bits))

    rslt = [sets.rational_out_overflow(val * sets.minimum_step_as_rational) for val in stimuli]
    assert rslt == check


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("frac_bits", [0, 1, 2, 3])
@pytest.mark.parametrize("is_signed", [False, True])
def test_rational_underflow(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    sets = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)

    stimuli = list()
    stimuli.extend(sets.minimum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.minimum_as_integer + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer - sets.total_bits + val for val in range(sets.total_bits))
    stimuli.extend(sets.maximum_as_integer + 1 + val for val in range(sets.total_bits))

    check = list()
    check.extend(True for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))
    check.extend(False for _ in range(sets.total_bits))

    rslt = [sets.rational_out_underflow(val * sets.minimum_step_as_rational) for val in stimuli]
    assert rslt == check
