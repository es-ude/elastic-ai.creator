import pytest
import torch

from .accumulator import accumulator_bits


@pytest.mark.parametrize("products,expected", [(1, 18), (3, 20), (100, 25)])
def test_reduction_width(products, expected):
    assert accumulator_bits(18, products) == expected


@pytest.mark.parametrize("bias", [-524288, 524288])
def test_bias_width_can_exceed_product_width(bias):
    assert accumulator_bits(18, bias=torch.tensor([bias], dtype=torch.int32)) == 22


def test_rejects_accumulator_wider_than_storage():
    with pytest.raises(ValueError, match="int64"):
        accumulator_bits(64, bias=torch.tensor([1], dtype=torch.int32))
