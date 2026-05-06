import pytest
import torch

import elasticai.creator.nn.fixed_point as nn_fxp
from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn import Sequential


class SimplePReLU2(torch.nn.Module):
    def __init__(self, total_width: int, frac_width: int, init: float):
        super().__init__()
        self.model = Sequential(
            nn_fxp.Linear(
                in_features=10,
                out_features=10,
                total_bits=total_width,
                frac_bits=frac_width,
            ),
            nn_fxp.PReLU2(total_bits=total_width, frac_bits=frac_width, init=init),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@pytest.mark.slow
@pytest.mark.parametrize("total_bits, frac_bits, init", [(4, 3, 0.25)])
def test_trainable_layer_prelu2(total_bits: int, frac_bits: int, init: float) -> None:
    params = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    vrange = (params.minimum_as_rational, params.maximum_as_rational)

    stimuli = torch.rand((4, 10)) * (vrange[1] - vrange[0]) + vrange[0]
    model = SimplePReLU2(total_bits, frac_bits, init)
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())

    for epoch in range(2):
        y = model(stimuli).squeeze()
        loss = criterion(y, stimuli.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
