from collections.abc import Callable
from typing import Optional

import torch


class _LSTMCellBase(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        linear_factory: Callable[[int, int, bool], torch.nn.Linear],
        sigmoid_factory: Callable[[], torch.nn.Hardsigmoid],
        tanh_factory: Callable[[], torch.nn.Hardtanh],
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.sigmoid = sigmoid_factory()
        self.tanh = tanh_factory()

        self.linear_f = linear_factory(input_size + hidden_size, hidden_size, bias)
        self.linear_i = linear_factory(input_size + hidden_size, hidden_size, bias)
        self.linear_g = linear_factory(input_size + hidden_size, hidden_size, bias)
        self.linear_o = linear_factory(input_size + hidden_size, hidden_size, bias)

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            zeros = torch.zeros(*(*x.shape[:-1], self.hidden_size), dtype=x.dtype)
            h_prev, c_prev = torch.clone(zeros), torch.clone(zeros)
        else:
            h_prev, c_prev = state

        inputs = torch.cat((x, h_prev), dim=1)

        f = self.sigmoid(self.linear_f(inputs))
        i = self.sigmoid(self.linear_i(inputs))
        g = self.tanh(self.linear_g(inputs))
        o = self.sigmoid(self.linear_o(inputs))

        c = f * c_prev + i * g
        h = o * self.tanh(c)

        return h, c

    def quantized_forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("The quantized_forward function is not implemented.")
