from collections.abc import Callable
from functools import partial
from typing import Optional

import torch

from elasticai.creator.vhdl.number_representations import (
    FixedPointFactory,
    fixed_point_params_from_factory,
)
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
        self.mul_op = mul_op
        self.add_op = add_op

        self.linear_ih = linear_factory(
            in_features=input_size, out_features=hidden_size * 4, bias=bias
        )
        self.linear_hh = linear_factory(
            in_features=hidden_size, out_features=hidden_size * 4, bias=bias
        )
        self.sigmoid = sigmoid_factory()
        self.tanh = tanh_factory()

    def _fake_quant(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.input_dequant(self.input_quant(x)) for x in inputs)

    def _initialize_previous_state(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            zeros = torch.zeros(*(*x.shape[:-1], self.hidden_size), dtype=x.dtype)
            return torch.clone(zeros), torch.clone(zeros)
        return state

    def forward(
        self, x: torch.Tensor, state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = self._initialize_previous_state(x, state)
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

        c = self.add_op(self.mul_op(f, c_prev), self.mul_op(i, g))
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
            input_quant=self.quant,
            input_dequant=self.dequant,
        )

    def quantized_forward(
        self, x: torch.Tensor, state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_bits, frac_bits = fixed_point_params_from_factory(
            self.fixed_point_factory
        )

        def fp_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return ((a * b) / (1 << frac_bits)).int().float()

        def clamp_overflowing_values(a: torch.Tensor) -> torch.Tensor:
            largest_fp_int = 2 ** (total_bits - 1) - 1
            return a.clamp(-largest_fp_int, largest_fp_int)

        h_prev, c_prev = self._initialize_previous_state(x, state)

        pred_ii, pred_if, pred_ig, pred_io = torch.split(
            self.linear_ih.quantized_forward(x), self.hidden_size, dim=1
        )
        pred_hi, pred_hf, pred_hg, pred_ho = torch.split(
            self.linear_hh.quantized_forward(h_prev), self.hidden_size, dim=1
        )

        i = self.sigmoid.quantized_forward(pred_ii + pred_hi)
        f = self.sigmoid.quantized_forward(pred_if + pred_hf)
        g = self.tanh.quantized_forward(pred_ig + pred_hg)
        o = self.sigmoid.quantized_forward(pred_io + pred_ho)

        c = self.add_op(fp_mul(f, c_prev), fp_mul(i, g))
        h = fp_mul(o, self.tanh.quantized_forward(c))

        h_clamped = clamp_overflowing_values(h)
        c_clamped = clamp_overflowing_values(c)

        return h_clamped, c_clamped
