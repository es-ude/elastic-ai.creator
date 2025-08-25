import pytest
import torch

from elasticai.creator.nn import fixed_point as nn_creator


@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps",
    [(4, 3, 4), (6, 4, 32), (8, 5, 64), (10, 6, 16), (12, 7, 128), (12, 11, 64)],
)
def test_hard_tanh_compared_torch(
    total_bits: int, frac_bits: int, num_steps: int, val_range=(-1.0, +1.0)
) -> None:
    fxp = FxpArithmetic(total_bits=total_bits, frac_bits=frac_bits)
    vrange = (fxp.minimum_as_rational, fxp.maximum_as_rational)
    stimulus = torch.arange(
        start=vrange[0], end=vrange[1], step=fxp.minimum_step_as_rational
    )

    act0 = torch.nn.Hardtanh(min_val=val_range[0], max_val=val_range[1])
    out0 = act0(stimulus)
    act1 = nn_creator.HardTanh(
        total_bits=total_bits,
        frac_bits=frac_bits,
        min_val=val_range[0],
        max_val=val_range[1],
    )
    out1 = act1(stimulus)
    assert float(sum(abs(out1 - out0))) < 1e-9
    assert float(abs(sum(out1 - out0))) < 1e-6
