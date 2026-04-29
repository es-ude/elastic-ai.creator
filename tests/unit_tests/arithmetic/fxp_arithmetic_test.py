import pytest

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_cut_float_to_integer_signed(total_bits: int, frac_bits: int) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    chck = [
        config.config.minimum_as_integer,
        -1,
        0,
        +1,
        config.config.maximum_as_integer,
    ]
    stimuli = [
        config.minimum_as_rational,
        -config.config.minimum_step_as_rational,
        0.0,
        config.config.minimum_step_as_rational,
        config.maximum_as_rational,
    ]
    rslt = [config.cut_as_integer(val) for val in stimuli]
    assert rslt == chck


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_round_float_to_integer_signed(total_bits: int, frac_bits: int) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    chck = [
        config.config.minimum_as_integer,
        -1,
        0,
        +1,
        config.config.maximum_as_integer,
    ]
    stimuli = [
        config.config.minimum_as_rational,
        -config.config.minimum_step_as_rational,
        0.0,
        config.config.minimum_step_as_rational,
        config.config.maximum_as_rational,
    ]
    rslt = [config.round_to_integer(val) for val in stimuli]
    assert rslt == chck


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_cut_float_to_integer_unsigned(total_bits: int, frac_bits: int) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=False)
    )
    chck = [
        config.config.minimum_as_integer,
        1,
        config.config.maximum_as_integer - 1,
        config.config.maximum_as_integer,
    ]
    stimuli = [
        config.config.minimum_as_rational,
        config.config.minimum_step_as_rational,
        config.config.maximum_as_rational - config.config.minimum_step_as_rational,
        config.config.maximum_as_rational,
    ]
    rslt = [config.cut_as_integer(val) for val in stimuli]
    assert rslt == chck


@pytest.mark.parametrize(
    "total_bits, frac_bits, signed, val_in, val_out",
    [
        (2, 1, True, 1, 0.5),
        (2, 1, True, 0, 0.0),
        (3, 2, True, -4, -1.0),
        (3, 2, True, 3, 0.75),
        (2, 1, False, 1, 0.5),
        (2, 1, False, 0, 0.0),
        (3, 2, False, 4, 1.0),
        (3, 2, False, 7, 1.75),
    ],
)
def test_integer_to_rational(
    total_bits: int, frac_bits: int, signed: bool, val_in: int, val_out: float
) -> None:
    result = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=signed)
    ).as_rational(val_in)
    assert result == val_out
