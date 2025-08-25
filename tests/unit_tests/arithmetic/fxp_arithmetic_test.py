import pytest
import torch

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from tests.tensor_test_case import assertTensorEqual


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
def test_cut_tensor_to_integer_signed(total_bits: int, frac_bits: int) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
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
    rslt = config.cut_as_integer(stimuli)
    assertTensorEqual(rslt, chck)


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_cut_x_to_integer_signed(total_bits: int, frac_bits: int) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
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
def test_round_tensor_to_integer_signed(total_bits: int, frac_bits: int) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
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
def test_round_x_to_integer_signed(total_bits: int, frac_bits: int) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
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


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_cut_tensor_to_integer_unsigned(total_bits: int, frac_bits: int) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=False)
    )
    chck = [
        config.config.minimum_as_integer,
        1,
        config.config.maximum_as_integer - 1,
        config.config.maximum_as_integer,
    ]
    stimuli = torch.Tensor(
        [
            config.config.minimum_as_rational,
            config.config.minimum_step_as_rational,
            config.config.maximum_as_rational - config.config.minimum_step_as_rational,
            config.config.maximum_as_rational,
        ]
    )
    rslt = config.cut_as_integer(stimuli)
    assertTensorEqual(rslt, chck)


@pytest.mark.parametrize("total_bits, frac_bits", [(2, 2), (4, 3), (8, 4)])
def test_round_tensor_to_integer_unsigned(total_bits: int, frac_bits: int) -> None:
    config = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=False)
    )

    chck = [
        config.config.minimum_as_integer,
        1,
        config.config.maximum_as_integer - 1,
        config.config.maximum_as_integer,
    ]
    stimuli = torch.Tensor(
        [
            config.config.minimum_as_rational,
            config.config.minimum_step_as_rational,
            config.config.maximum_as_rational - config.config.minimum_step_as_rational,
            config.config.maximum_as_rational,
        ]
    )
    rslt = config.round_to_integer(stimuli)
    assertTensorEqual(rslt, chck)


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
