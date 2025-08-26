import pytest
import torch

import elasticai.creator.nn.fixed_point as nn_fxp
from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn import Sequential


class SimpleLinear(torch.nn.Module):
    def __init__(self, total_width: int, frac_width: int):
        super().__init__()
        self.model = Sequential(
            nn_fxp.Linear(
                in_features=10,
                out_features=10,
                total_bits=total_width,
                frac_bits=frac_width,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@pytest.mark.slow
@pytest.mark.parametrize("total_bits, frac_bits, num_steps", [(4, 3, 4)])
def test_trainable_layer_linear(
    total_bits: int, frac_bits: int, num_steps: int
) -> None:
    params = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    vrange = (params.minimum_as_rational, params.maximum_as_rational)

    stimuli = torch.rand((4, 10)) * (vrange[1] - vrange[0]) + vrange[0]
    model = SimpleLinear(total_bits, frac_bits)
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())

    for epoch in range(2):
        y = model(stimuli).squeeze()
        loss = criterion(y, torch.zeros_like(stimuli).float())
        optim.zero_grad()
        loss.backward()
        optim.step()


class SimpleBatchNormLinear(torch.nn.Module):
    def __init__(self, total_width: int, frac_width: int):
        super().__init__()
        self.model = Sequential(
            nn_fxp.BatchNormedLinear(
                in_features=10,
                out_features=10,
                total_bits=total_width,
                frac_bits=frac_width,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@pytest.mark.slow
@pytest.mark.parametrize("total_bits, frac_bits, num_steps", [(4, 3, 4)])
def test_trainable_layer_batchnorm_linear(
    total_bits: int, frac_bits: int, num_steps: int
) -> None:
    params = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    vrange = (params.minimum_as_rational, params.maximum_as_rational)

    stimuli = torch.rand((4, 10)) * (vrange[1] - vrange[0]) + vrange[0]

    model = SimpleBatchNormLinear(total_bits, frac_bits)
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())

    for epoch in range(2):
        y = model(stimuli).squeeze()
        loss = criterion(y, torch.zeros_like(stimuli).float())
        optim.zero_grad()
        loss.backward()
        optim.step()
