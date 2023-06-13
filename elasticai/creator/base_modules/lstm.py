from typing import Optional, Protocol

import torch

from .lstm_cell import LSTMCell


class LayerFactory(Protocol):
    def lstm(self, input_size: int, hidden_size: int, bias: bool) -> LSTMCell:
        ...


class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size,
        bias: bool,
        batch_first: bool,
        layers: LayerFactory,
    ) -> None:
        super().__init__()
        self.cell = layers.lstm(
            input_size=input_size, hidden_size=hidden_size, bias=bias
        )
        self.batch_first = batch_first

    @property
    def hidden_size(self) -> int:
        return self.cell.hidden_size

    @property
    def input_size(self) -> int:
        return self.cell.input_size

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
        # TODO: check whether unsqueeze dimension is actually consistent with self.batch_first being true or false
        hidden_state, cell_state = state[0].unsqueeze(0), state[1].unsqueeze(0)
        return result, (hidden_state, cell_state)
