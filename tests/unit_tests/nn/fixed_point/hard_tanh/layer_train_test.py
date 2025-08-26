import pytest
import torch

import elasticai.creator.nn.fixed_point as nn
from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn import Sequential


class SimpleHardTanh(torch.nn.Module):
    def __init__(self, total_width: int, frac_width: int, vrange: tuple[float, float]):
        super().__init__()
        self.model = Sequential(
            nn.Linear(
                in_features=10,
                out_features=10,
                total_bits=total_width,
                frac_bits=frac_width,
            ),
            nn.HardTanh(
                total_bits=total_width,
                frac_bits=frac_width,
                min_val=vrange[0],
                max_val=vrange[1],
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@pytest.mark.slow
@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps, yrange", [(4, 3, 4, (-1.0, +1.0))]
)
def test_trainable_layer_hardtanh(
    total_bits: int, frac_bits: int, num_steps: int, yrange: tuple[float, float]
) -> None:
    params = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    vrange = (params.minimum_as_rational, params.maximum_as_rational)

    stimuli = torch.rand((4, 10)) * (vrange[1] - vrange[0]) + vrange[0]
    model = SimpleHardTanh(total_bits, frac_bits, yrange)
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())

    for epoch in range(2):
        y = model(stimuli).squeeze()
        loss = criterion(y, stimuli.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
