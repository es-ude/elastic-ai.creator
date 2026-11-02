import warnings

import pytest
import torch

from .accumulator import accumulator_bits


@pytest.mark.parametrize("products,expected", [(1, 18), (3, 20), (100, 25)])
def test_reduction_width(products, expected):
    assert accumulator_bits(18, products) == expected


@pytest.mark.parametrize("bias", [-524288, 524288])
def test_bias_width_can_exceed_product_width(bias):
    assert accumulator_bits(18, bias=torch.tensor([bias], dtype=torch.int32)) == 22


@pytest.mark.parametrize("bits", [33, 64, 65])
def test_caps_accumulator_width_and_warns(bits):
    with pytest.warns(RuntimeWarning, match="Overflow could occur"):
        assert accumulator_bits(bits) == 32


def test_32_bit_accumulator_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert accumulator_bits(32) == 32
    assert not caught


def test_bias_can_trigger_accumulator_width_warning():
    with pytest.warns(RuntimeWarning, match="calculated accumulator width is 33 bits"):
        assert accumulator_bits(32, bias=torch.tensor([1], dtype=torch.int32)) == 32
