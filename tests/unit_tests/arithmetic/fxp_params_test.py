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


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, val_in, expected",
    [
        (2, 1, True, 2, True),
        (2, 1, True, 1, False),
        (3, 1, True, 4, True),
        (3, 1, True, 3, False),
        (2, 1, False, 4, True),
        (2, 1, False, 3, False),
        (3, 1, False, 8, True),
        (3, 1, False, 7, False),
    ],
)
def test_integer_upper_limit(
    total_bits: int, frac_bits: int, signed: bool, val_in: int, expected: bool
) -> None:
    result = FxpParams(
        total_bits=total_bits, frac_bits=frac_bits, signed=signed
    ).integer_out_overflow(val_in)
    assert result == expected


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, val_in, expected",
    [
        (2, 1, True, -3, True),
        (2, 1, True, -2, False),
        (3, 1, True, -5, True),
        (3, 1, True, -4, False),
        (2, 1, False, -1, True),
        (2, 1, False, 0, False),
        (3, 1, False, -1, True),
        (3, 1, False, 0, False),
    ],
)
def test_integer_downer_limit(
    total_bits: int, frac_bits: int, signed: bool, val_in: int, expected: bool
) -> None:
    result = FxpParams(
        total_bits=total_bits, frac_bits=frac_bits, signed=signed
    ).integer_out_underflow(val_in)
    assert result == expected


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, val_in, expected",
    [
        (2, 1, True, -3, True),
        (2, 1, True, 2, True),
        (2, 1, True, -2, False),
        (2, 1, True, 1, False),
        (3, 1, True, -5, True),
        (3, 1, True, -4, False),
        (3, 1, True, 3, False),
        (3, 1, True, 4, True),
        (2, 1, False, -1, True),
        (2, 1, False, 0, False),
        (3, 1, False, -1, True),
        (3, 1, False, 0, False),
    ],
)
def test_integer_out_of_bounds(
    total_bits: int, frac_bits: int, signed: bool, val_in: int, expected: bool
) -> None:
    result = FxpParams(
        total_bits=total_bits, frac_bits=frac_bits, signed=signed
    ).integer_out_of_bounds(val_in)
    assert result == expected


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, val_in, expected",
    [
        (2, 1, True, 1.0, True),
        (2, 1, True, 0.5, False),
        (3, 1, True, 2.0, True),
        (3, 1, True, 1.5, False),
        (2, 1, True, 1.0, True),
        (2, 1, True, 0.5, False),
        (3, 1, True, 2.0, True),
        (3, 1, True, 1.5, False),
        (2, 1, False, 2.0, True),
        (2, 1, False, 1.5, False),
        (3, 1, False, 4.0, True),
        (3, 1, False, 3.5, False),
    ],
)
def test_rational_upper_limit(
    total_bits: int, frac_bits: int, signed: bool, val_in: int, expected: bool
) -> None:
    result = FxpParams(
        total_bits=total_bits, frac_bits=frac_bits, signed=signed
    ).rational_out_overflow(val_in)
    assert result == expected


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, val_in, expected",
    [
        (2, 1, True, -1.5, True),
        (2, 1, True, -1.0, False),
        (3, 1, True, -2.5, True),
        (3, 1, True, -2.0, False),
        (2, 1, False, -0.5, True),
        (2, 1, False, 0.0, False),
        (3, 1, False, -0.5, True),
        (3, 1, False, 0.0, False),
    ],
)
def test_rational_downer_limit(
    total_bits: int, frac_bits: int, signed: bool, val_in: int, expected: bool
) -> None:
    result = FxpParams(
        total_bits=total_bits, frac_bits=frac_bits, signed=signed
    ).rational_out_underflow(val_in)
    assert result == expected


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, val_in, expected",
    [
        (2, 1, True, -1.5, True),
        (2, 1, True, -1.0, False),
        (3, 1, True, -2.5, True),
        (3, 1, True, -2.0, False),
        (2, 1, True, 1.0, True),
        (2, 1, True, 0.5, False),
        (3, 1, True, 2.0, True),
        (3, 1, True, 1.5, False),
        (2, 1, False, -0.5, True),
        (2, 1, False, 0.0, False),
        (3, 1, False, -0.5, True),
        (3, 1, False, 0.0, False),
    ],
)
def test_rational_out_of_bounds(
    total_bits: int, frac_bits: int, signed: bool, val_in: int, expected: bool
) -> None:
    result = FxpParams(
        total_bits=total_bits, frac_bits=frac_bits, signed=signed
    ).rational_out_of_bounds(val_in)
    assert result == expected
