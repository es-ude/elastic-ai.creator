import warnings

import pytest

from ..quant_utils.QParams import QuantScheme
from .accumulator import accumulator_bits


@pytest.mark.parametrize(
    "input_bits,weight_bits,reduction_shape,expected",
    [(8, 8, (), 18), (8, 8, (3,), 20), (8, 8, (100,), 25), (6, 4, (3, 5), 16)],
)
def test_reduction_width(input_bits, weight_bits, reduction_shape, expected):
    assert accumulator_bits(input_bits, weight_bits, reduction_shape) == expected


@pytest.mark.parametrize(
    "input_scheme,weight_scheme,expected",
    [
        (QuantScheme.SYMMETRIC_SIGNED, QuantScheme.SYMMETRIC_SIGNED, 18),
        (QuantScheme.SYMMETRIC_SIGNED, QuantScheme.ASYMMETRIC_SIGNED, 19),
        (QuantScheme.ASYMMETRIC_SIGNED, QuantScheme.SYMMETRIC_SIGNED, 19),
        (QuantScheme.ASYMMETRIC_SIGNED, QuantScheme.ASYMMETRIC_SIGNED, 20),
        (QuantScheme.SYMMETRIC_UNSIGNED, QuantScheme.SYMMETRIC_SIGNED, 18),
        (QuantScheme.SYMMETRIC_UNSIGNED, QuantScheme.SYMMETRIC_UNSIGNED, 19),
    ],
)
def test_accumulator_width_accounts_for_quantization_scheme(
    input_scheme, weight_scheme, expected
):
    assert (
        accumulator_bits(
            8, 8, (3,), input_scheme=input_scheme, weight_scheme=weight_scheme
        )
        == expected
    )


@pytest.mark.parametrize(
    "input_bits,weight_bits,reduction_shape",
    [(15, 15, (2,)), (31, 31, ()), (31, 32, ())],
)
def test_caps_accumulator_width_and_warns(input_bits, weight_bits, reduction_shape):
    with pytest.warns(RuntimeWarning, match="Overflow could occur"):
        assert accumulator_bits(input_bits, weight_bits, reduction_shape) == 32


def test_32_bit_accumulator_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert accumulator_bits(15, 15) == 32
    assert not caught
