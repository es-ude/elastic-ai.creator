from collections.abc import Callable
from typing import Optional

import torch.nn


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

    def forward(
        self, x: torch.Tensor, state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def quantized_forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("The quantized_forward function is not implemented.")
