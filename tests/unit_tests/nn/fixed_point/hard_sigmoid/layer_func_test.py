import pytest
import torch

from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn import fixed_point as nn_creator


@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps",
    [(4, 3, 4), (6, 4, 32), (8, 5, 64), (10, 6, 16), (12, 7, 128), (12, 11, 64)],
)
def test_hard_sigmoid_compared_torch(
    total_bits: int, frac_bits: int, num_steps: int
) -> None:
    params = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    vrange = (params.minimum_as_rational, params.maximum_as_rational)
    stimulus = torch.arange(
        start=vrange[0], end=vrange[1], step=params.minimum_step_as_rational
    )

    act0 = torch.nn.Hardsigmoid()
    out0 = act0(stimulus)
    act1 = nn_creator.HardSigmoid(total_bits=total_bits, frac_bits=frac_bits)
    out1 = act1(stimulus)
    assert float(sum(abs(out1 - out0))) < 1e-6
    assert float(abs(sum(out1 - out0))) < 1e-6
