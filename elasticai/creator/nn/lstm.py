from collections.abc import Callable
from functools import partial
from typing import Optional, cast

import torch

from elasticai.creator.nn.arithmetics import FixedPointArithmetics
from elasticai.creator.nn.hard_sigmoid import HardSigmoid
from elasticai.creator.nn.hard_tanh import HardTanh
from elasticai.creator.nn.lstm_cell import LSTMCell
from elasticai.creator.vhdl.number_representations import FixedPointFactory


class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size,
        bias: bool,
        batch_first: bool,
        lstm_cell_factory: Callable[..., torch.nn.Module],
    ) -> None:
        super().__init__()
        self.cell = lstm_cell_factory(
            input_size=input_size, hidden_size=hidden_size, bias=bias
        )
        self.batch_first = batch_first

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batched = x.dim() == 3

        if batched and self.batch_first:
            x = torch.stack(torch.unbind(x), dim=1)

        if state is not None:
            state = state[0].squeeze(0), state[1].squeeze(0)

        inputs = torch.unbind(x, dim=0)

        outputs = []
        for i in range(len(inputs)):
            hidden_state, cell_state = self.cell(inputs[i], state)
            state = (hidden_state, cell_state)
            outputs.append(hidden_state)

        if state is None:
            raise RuntimeError("Number of samples must be larger than 0.")

        result = torch.stack(outputs, dim=1 if batched and self.batch_first else 0)
        hidden_state, cell_state = state[0].unsqueeze(0), state[1].unsqueeze(0)

        return result, (hidden_state, cell_state)


class FixedPointLSTMWithHardActivations(LSTM):
    def __init__(
        self,
        fixed_point_factory: FixedPointFactory,
        input_size: int,
        hidden_size: int,
        batch_first: bool,
        bias: bool = True,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            batch_first=batch_first,
            lstm_cell_factory=partial(
                LSTMCell,
                arithmetics=FixedPointArithmetics(
                    fixed_point_factory=fixed_point_factory
                ),
                sigmoid_factory=HardSigmoid,
                tanh_factory=HardTanh,
            ),
        )

    @property
    def fixed_point_factory(self) -> FixedPointFactory:
        return cast(FixedPointArithmetics, self.cell.ops).fixed_point_factory
