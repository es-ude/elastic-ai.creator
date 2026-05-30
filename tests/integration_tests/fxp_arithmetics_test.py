from functools import partial

import pytest
import torch
from torch.testing import assert_close

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.arithmetic.fxp_params import ConvertableToFixedPointValues

assertTensorEqual = partial(assert_close, rtol=0, atol=0)


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
@pytest.mark.parametrize("is_signed", [False, True])
def test_cut_tensor_to_integer(
    total_bits: int, frac_bits: int, is_signed: bool
) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    )
    stimuli = torch.Tensor(
        [
            config.config.minimum_as_rational,
            config.config.minimum_step_as_rational,
            config.config.maximum_as_rational - config.config.minimum_step_as_rational,
            config.config.maximum_as_rational,
        ]
    )
    chck = torch.tensor(
        [
            config.config.minimum_as_integer,
            1,
            config.config.maximum_as_integer - 1,
            config.config.maximum_as_integer,
        ],
        dtype=stimuli.dtype,
    )
    rslt = config.cut_as_integer(stimuli)
    assertTensorEqual(rslt, chck)


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
@pytest.mark.parametrize("is_signed", [False, True])
def test_cut_x_to_integer(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    )

    stimuli_tensor = torch.Tensor(
        [
            config.config.minimum_as_rational,
            -config.config.minimum_step_as_rational,
            0,
            config.config.minimum_step_as_rational,
            config.config.maximum_as_rational,
        ]
    )
    stimuli_float = [
        config.config.minimum_as_rational,
        -config.config.minimum_step_as_rational,
        0.0,
        config.config.minimum_step_as_rational,
        config.config.maximum_as_rational,
    ]
    rslt_tensor = config.cut_as_integer(stimuli_tensor).tolist()
    rslt_float = [config.cut_as_integer(val) for val in stimuli_float]
    assert rslt_tensor == rslt_float


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
@pytest.mark.parametrize("is_signed", [False, True])
def test_round_tensor_to_integer(
    total_bits: int, frac_bits: int, is_signed: bool
) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    )

    chck = torch.Tensor(
        [config.config.minimum_as_integer, -1, 0, +1, config.config.maximum_as_integer]
    )
    stimuli = torch.Tensor(
        [
            config.config.minimum_as_rational,
            -config.config.minimum_step_as_rational,
            0.0,
            config.config.minimum_step_as_rational,
            config.config.maximum_as_rational,
        ]
    )
    rslt = config.round_to_integer(stimuli)
    assertTensorEqual(rslt, chck)


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
@pytest.mark.parametrize("is_signed", [False, True])
def test_round_x_to_integer(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    )

    stimuli_tensor = torch.Tensor(
        [
            config.config.minimum_as_rational,
            -config.config.minimum_step_as_rational,
            0.0,
            config.config.minimum_step_as_rational,
            config.config.maximum_as_rational,
        ]
    )
    stimuli_float = [
        config.config.minimum_as_rational,
        -config.config.minimum_step_as_rational,
        0.0,
        config.config.minimum_step_as_rational,
        config.config.maximum_as_rational,
    ]
    rslt_tensor = config.round_to_integer(stimuli_tensor).tolist()
    rslt_float = [config.round_to_integer(val) for val in stimuli_float]
    assert rslt_tensor == rslt_float


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("frac_bits", [0, 1, 2, 3])
@pytest.mark.parametrize("is_signed", [False, True])
def test_clamp_float(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    sets = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    config = FxpArithmetic(fxp_params=sets)

    stimuli = list()
    stimuli.extend(
        sets.minimum_as_rational
        - (sets.total_bits - val) * sets.minimum_step_as_rational
        for val in range(sets.total_bits)
    )
    stimuli.extend(
        sets.minimum_as_rational + val * sets.minimum_step_as_rational
        for val in range(sets.total_bits)
    )
    stimuli.extend(
        sets.maximum_as_rational
        - (sets.total_bits - val) * sets.minimum_step_as_rational
        for val in range(sets.total_bits)
    )
    stimuli.extend(
        sets.maximum_as_rational + val * sets.minimum_step_as_rational
        for val in range(sets.total_bits)
    )

    check = list()
    check.extend(sets.minimum_as_rational for _ in range(sets.total_bits))
    check.extend(
        sets.minimum_as_rational + val * sets.minimum_step_as_rational
        for val in range(sets.total_bits)
    )
    check.extend(
        sets.maximum_as_rational
        - (sets.total_bits - val) * sets.minimum_step_as_rational
        for val in range(sets.total_bits)
    )
    check.extend(sets.maximum_as_rational for _ in range(sets.total_bits))

    result = [config.clamp(val) for val in stimuli]
    assert result == check

    result = config.clamp(torch.asarray(stimuli)).tolist()
    assert result == check


@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
@pytest.mark.parametrize("frac_bits", [0, 1, 2, 3])
@pytest.mark.parametrize("is_signed", [False, True])
def test_clamp_integer(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    sets = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    config = FxpArithmetic(fxp_params=sets)

    stimuli = list()
    stimuli.extend(
        sets.minimum_as_integer - sets.total_bits + val
        for val in range(sets.total_bits)
    )
    stimuli.extend(sets.minimum_as_integer + val for val in range(sets.total_bits))
    stimuli.extend(
        sets.maximum_as_integer - sets.total_bits + val
        for val in range(sets.total_bits)
    )
    stimuli.extend(sets.maximum_as_integer + val for val in range(sets.total_bits))

    check = list()
    check.extend(sets.minimum_as_integer for _ in range(sets.total_bits))
    check.extend(sets.minimum_as_integer + val for val in range(sets.total_bits))
    check.extend(
        sets.maximum_as_integer - sets.total_bits + val
        for val in range(sets.total_bits)
    )
    check.extend(sets.maximum_as_integer for _ in range(sets.total_bits))

    result = [config.clamp(val) for val in stimuli]
    assert result == check

    if frac_bits == 0:
        result = config.clamp(torch.asarray(stimuli)).tolist()
        assert result == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, is_signed, val_in, expected",
    [
        (2, 1, True, -1, True),
        (2, 1, True, 0, False),
        (2, 1, True, 1, True),
        (2, 1, False, 0, False),
        (2, 1, False, 2, True),
        (2, 1, False, 3, False),
        (8, 4, True, -128, True),
        (8, 4, True, -60, False),
        (8, 4, False, 1, True),
        (8, 4, False, 254, False),
    ],
)
def test_is_power_2(
    total_bits: int, frac_bits: int, is_signed: bool, val_in, expected: bool
) -> None:
    sets = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    )

    results = sets.is_power_of_2(val_in)
    assert results == expected

    data_in = torch.asarray([val_in for _ in range(10)])
    check = [expected for _ in range(10)]
    results = sets.is_power_of_2(data_in).tolist()
    assert results == check


@pytest.mark.parametrize(
    "total_bits, frac_bits, is_signed, val_in, expected",
    [
        (2, 1, True, -1, 3),
        (2, 1, True, -0.5, 1.5),
        (2, 1, True, 1, 1),
        (2, 1, False, 1, 1),
        (2, 1, False, 1.0, 1.0),
        (8, 3, True, -8.125, 23.875),
        (8, 3, False, 1.0, 1.0),
    ],
)
def test_to_twos(
    total_bits: int,
    frac_bits: int,
    is_signed: bool,
    val_in: int | float,
    expected: int | float,
) -> None:
    sets = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    )

    result = sets.to_twos(val_in)
    assert result == expected

    data_in = torch.asarray([val_in for _ in range(10)])
    check = [expected for _ in range(10)]
    rslt = sets.to_twos(data_in).tolist()
    assert rslt == check


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
@pytest.mark.parametrize("is_signed", [False, True])
def test_cut_tensor_to_rational(
    total_bits: int, frac_bits: int, is_signed: bool
) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    )

    chck: ConvertableToFixedPointValues = (
        torch.Tensor(
            [
                config.config.minimum_as_integer,
                -1,
                0,
                +1,
                config.config.maximum_as_integer,
            ]
        )
        * config.config.minimum_step_as_rational
    )
    stimuli = torch.Tensor(
        [
            config.config.minimum_as_rational,
            -config.config.minimum_step_as_rational,
            0.0,
            config.config.minimum_step_as_rational,
            config.config.maximum_as_rational,
        ]
    )
    rslt = config.cut_as_rational(stimuli)
    assertTensorEqual(rslt, chck)


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
@pytest.mark.parametrize("is_signed", [False, True])
def test_cut_x_to_rational(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    )

    stimuli_tensor = torch.Tensor(
        [
            config.config.minimum_as_rational,
            -config.config.minimum_step_as_rational,
            0,
            config.config.minimum_step_as_rational,
            config.config.maximum_as_rational,
        ]
    )
    stimuli_float = [
        config.config.minimum_as_rational,
        -config.config.minimum_step_as_rational,
        0.0,
        config.config.minimum_step_as_rational,
        config.config.maximum_as_rational,
    ]
    rslt_tensor = config.cut_as_rational(stimuli_tensor).tolist()
    rslt_float = [config.cut_as_rational(val) for val in stimuli_float]
    assert rslt_tensor == rslt_float


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
@pytest.mark.parametrize("is_signed", [False, True])
def test_round_tensor_to_rational(
    total_bits: int, frac_bits: int, is_signed: bool
) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    )

    chck = (
        torch.Tensor(
            [
                config.config.minimum_as_integer,
                -1,
                0,
                +1,
                config.config.maximum_as_integer,
            ]
        )
        * config.config.minimum_step_as_rational
    )
    stimuli = torch.Tensor(
        [
            config.config.minimum_as_rational,
            -config.config.minimum_step_as_rational,
            0.0,
            config.config.minimum_step_as_rational,
            config.config.maximum_as_rational,
        ]
    )
    rslt = config.round_to_rational(stimuli)
    assertTensorEqual(rslt, chck)


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
@pytest.mark.parametrize("is_signed", [False, True])
def test_round_x_to_rational(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=is_signed)
    )

    stimuli = torch.tensor(
        [
            config.config.minimum_as_rational,
            config.config.minimum_step_as_rational,
            config.config.maximum_as_rational - config.config.minimum_step_as_rational,
            config.config.maximum_as_rational,
        ]
    )
    chck = torch.tensor(
        [
            config.config.minimum_as_rational,
            config.config.minimum_step_as_rational,
            (config.config.maximum_as_integer - 1)
            * config.config.minimum_step_as_rational,
            config.config.maximum_as_rational,
        ],
        dtype=stimuli.dtype,
    )
    rslt = config.round_to_rational(stimuli)
    assertTensorEqual(rslt, chck)
