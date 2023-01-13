from collections.abc import Callable
from functools import partial
from typing import Optional

import torch.nn

from elasticai.creator.nn.lstm_cell import FixedPointLSTMCell
from elasticai.creator.vhdl.number_representations import FixedPointFactory


class _LSTMBase(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size,
        bias: bool,
        batch_first: bool,
        lstm_cell_factory: Callable,
    ) -> None:
        super().__init__()
        self.cell = lstm_cell_factory(
            input_size=input_size, hidden_size=hidden_size, bias=bias
        )
        self.batch_first = batch_first

    def _do_forward(
        self,
        forward_func: Callable[
            [torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]],
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        ],
        x: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batched = x.dim() == 3

        if batched and self.batch_first:
            x = torch.stack(torch.unbind(x), dim=1)

        if state is not None:
            state = state[0].squeeze(0), state[1].squeeze(0)

        inputs = torch.unbind(x, dim=0)

        outputs = []
        for i in range(len(inputs)):
            hidden_state, cell_state = forward_func(inputs[i], state)
            state = (hidden_state, cell_state)
            outputs.append(hidden_state)

        result = torch.stack(outputs, dim=1 if batched and self.batch_first else 0)
        hidden_state, cell_state = state[0].unsqueeze(0), state[1].unsqueeze(0)

        return result, (hidden_state, cell_state)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self._do_forward(self.cell.__call__, x, state)

    def quantized_forward(
        self,
        x: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self._do_forward(self.cell.quantized_forward, x, state)


class FixedPointLSTM(_LSTMBase):
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
                FixedPointLSTMCell, fixed_point_factory=fixed_point_factory
            ),
        )
