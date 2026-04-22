import torch

from elasticai.creator_plugins.quantized_grads.fixed_point import FixedPointConfigV2
from elasticai.creator_plugins.quantized_grads.fixed_point.quantize_to_fixed_point import (
    _clamp,
    _round_to_fixed_point_hte,
    quantize_to_fxp_hte,
)


def test_round_determinstic_to_fixed_point():
    conf = FixedPointConfigV2(total_bits=8, frac_bits=2)
    x = torch.arange(-2, 2.1, step=0.1, dtype=torch.float32)

    actual = _round_to_fixed_point_hte(x, conf.resolution_per_int)
    expected = torch.Tensor(
        [
            -2.0,
            -2.0,
            -1.75,
            -1.75,
            -1.5,
            -1.5,
            -1.5,
            -1.25,
            -1.25,
            -1.0,
            -1.0,
            -1.0,
            -0.75,
            -0.75,
            -0.5,
            -0.5,
            -0.5,
            -0.25,
            -0.25,
            -0.0,
            0.0,
            0.0,
            0.25,
            0.25,
            0.5,
            0.5,
            0.5,
            0.75,
            0.75,
            1.0,
            1.0,
            1.0,
            1.25,
            1.25,
            1.5,
            1.5,
            1.5,
            1.75,
            1.75,
            2.0,
            2.0,
        ]
    )
    assert torch.equal(actual, expected)


def test_clamp_to_fixed_point():
    conf = FixedPointConfigV2(total_bits=4, frac_bits=2)
    x = torch.arange(-3, 4, dtype=torch.float32)

    actual = _clamp(
        number=x,
        minimum_as_rational=conf.minimum_as_rational,
        maximum_as_rational=conf.maximum_as_rational,
    )
    expected = torch.Tensor([-2.0, -2.0, -1.0, 0.0, 1.0, 1.75, 1.75])
    assert torch.equal(actual, expected)


def test_quantize_determinstic_to_fixed_point():
    conf = FixedPointConfigV2(total_bits=4, frac_bits=2)
    x = torch.arange(-3, 3.1, step=0.33, dtype=torch.float32)
    actual = quantize_to_fxp_hte(
        number=x,
        resolution_per_int=conf.resolution_per_int,
        minimum_as_rational=conf.minimum_as_rational,
        maximum_as_rational=conf.maximum_as_rational,
    )
    expected = torch.Tensor(
        [
            -2.0,
            -2.0,
            -2.0,
            -2.0,
            -1.75,
            -1.25,
            -1.0,
            -0.75,
            -0.25,
            -0.0,
            0.25,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            1.75,
            1.75,
            1.75,
        ]
    )
    assert torch.equal(actual, expected)
