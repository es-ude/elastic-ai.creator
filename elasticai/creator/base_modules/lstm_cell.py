from collections.abc import Callable
from typing import Any, Optional, Protocol

import torch

from .linear import Linear
from .math_operations import Add, MatMul, Mul, Quantize


class MathOperations(Quantize, Add, MatMul, Mul, Protocol):
    ...


class LSTMCell(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        operations: MathOperations,
        sigmoid_factory: Callable[[], torch.nn.Module],
        tanh_factory: Callable[[], torch.nn.Module],
        device: Any = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self._operations = operations

        self.linear_ih = Linear(
            in_features=input_size,
            out_features=hidden_size * 4,
            bias=bias,
            operations=operations,
            device=device,
        )
        self.linear_hh = Linear(
            in_features=hidden_size,
            out_features=hidden_size * 4,
            bias=bias,
            operations=operations,
            device=device,
        )
        self.sigmoid = sigmoid_factory()
        self.tanh = tanh_factory()

    def _initialize_previous_state(
        self, x: torch.Tensor, state: Optional[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            zeros = torch.zeros(*(*x.shape[:-1], self.hidden_size), dtype=x.dtype)
            return torch.clone(zeros), torch.clone(zeros)
        return state

    def forward(
        self, x: torch.Tensor, state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = self._initialize_previous_state(x, state)

        pred_ii, pred_if, pred_ig, pred_io = torch.split(
            self.linear_ih(x), self.hidden_size, dim=-1
        )
        pred_hi, pred_hf, pred_hg, pred_ho = torch.split(
            self.linear_hh(h_prev), self.hidden_size, dim=-1
        )

        i = self.sigmoid(self._operations.add(pred_ii, pred_hi))
        f = self.sigmoid(self._operations.add(pred_if, pred_hf))
        g = self.tanh(self._operations.add(pred_ig, pred_hg))
        o = self.sigmoid(self._operations.add(pred_io, pred_ho))

        c = self._operations.add(
            self._operations.mul(f, c_prev), self._operations.mul(i, g)
        )
        h = self._operations.mul(o, self.tanh(c))

        return h, c
