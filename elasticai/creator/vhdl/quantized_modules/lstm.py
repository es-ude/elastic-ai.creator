from collections.abc import Callable
from functools import partial
from typing import Any, Optional

import torch

from elasticai.creator.vhdl.number_representations import FixedPointFactory
from elasticai.creator.vhdl.quantized_modules import (
    FixedPointHardSigmoid,
    FixedPointLinear,
)
from elasticai.creator.vhdl.quantized_modules.autograd_functions import (
    FixedPointDequantFunction,
    FixedPointQuantFunction,
)
from elasticai.creator.vhdl.quantized_modules.hard_tanh import FixedPointHardTanh
from elasticai.creator.vhdl.quantized_modules.typing import OperationType, QuantType


class _LSTMCellBase(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        linear_factory: Callable,
        sigmoid_factory: Callable,
        tanh_factory: Callable,
        mul_op: OperationType = torch.mul,
        add_op: OperationType = torch.add,
        input_quant: QuantType = lambda x: x,
        input_dequant: QuantType = lambda x: x,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.input_quant = input_quant
        self.input_dequant = input_dequant
        self.matmul_op = mul_op
        self.add_op = add_op

        self.linear_ih = linear_factory(input_size, hidden_size * 4, bias)
        self.linear_hh = linear_factory(hidden_size, hidden_size * 4, bias)
        self.sigmoid = sigmoid_factory()
        self.tanh = tanh_factory()

    def _fake_quant(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.input_dequant(self.input_quant(x)) for x in inputs)

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            zeros = torch.zeros(*(*x.shape[:-1], self.hidden_size), dtype=x.dtype)
            h_prev, c_prev = torch.clone(zeros), torch.clone(zeros)
        else:
            h_prev, c_prev = state

        x, h_prev, c_prev = self._fake_quant(x, h_prev, c_prev)

        pred_ii, pred_if, pred_ig, pred_io = torch.split(
            self.linear_ih(x), self.hidden_size, dim=1
        )
        pred_hi, pred_hf, pred_hg, pred_ho = torch.split(
            self.linear_hh(h_prev), self.hidden_size, dim=1
        )

        i = self.sigmoid(pred_ii + pred_hi)
        f = self.sigmoid(pred_if + pred_hf)
        g = self.tanh(pred_ig + pred_hg)
        o = self.sigmoid(pred_io + pred_ho)

        c = self.add_op(self.matmul_op(f, c_prev), self.matmul_op(i, g))
        h = o * self.tanh(c)

        return h, c

    def quantized_forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("The quantized_forward function is not implemented.")


class FixedPointLSTMCell(_LSTMCellBase):
    def __init__(
        self,
        fixed_point_factory: FixedPointFactory,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        self.fixed_point_factory = fixed_point_factory
        self.quant = lambda x: FixedPointQuantFunction.apply(x, fixed_point_factory)
        self.dequant = lambda x: FixedPointDequantFunction.apply(x, fixed_point_factory)
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            linear_factory=partial(
                FixedPointLinear, fixed_point_factory=fixed_point_factory
            ),
            sigmoid_factory=partial(
                FixedPointHardSigmoid, fixed_point_factory=fixed_point_factory
            ),
            tanh_factory=partial(
                FixedPointHardTanh, fixed_point_factory=fixed_point_factory
            ),
            input_quant=self.input_quant,
            input_dequant=self.input_dequant,
        )

    def quantized_forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
