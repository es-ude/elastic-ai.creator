from collections import OrderedDict
from typing import cast

import torch

from elasticai.creator.base_modules.hard_sigmoid import HardSigmoid
from elasticai.creator.base_modules.hard_tanh import HardTanh
from elasticai.creator.base_modules.lstm import LSTM
from elasticai.creator.base_modules.lstm_cell import LSTMCell
from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.nn.fixed_point_arithmetics import FixedPointArithmetics
from elasticai.creator.nn.two_complement_fixed_point_config import FixedPointConfig
from elasticai.creator.nn.vhdl.fp_linear_1d import FPLinear1d
from elasticai.creator.nn.vhdl.fp_linear_1d.design import (
    FPLinear1d as _FPLinear1dDesign,
)
from elasticai.creator.nn.vhdl.lstm.design.fp_lstm_cell import FPLSTMCell
from elasticai.creator.nn.vhdl.lstm.design.lstm import LSTMNetworkDesign
from elasticai.creator.nn.vhdl.module import Module


class LSTMNetwork(torch.nn.Module, Module):
    def __init__(self, layers: list[torch.nn.Module]):
        super().__init__()
        self.lstm = layers[0]
        self.layers = torch.nn.Sequential(
            OrderedDict(
                {f"fp_linear_1d_{i}": layer for i, layer in enumerate(layers[1:])}
            )
        )

    def translate(self) -> LSTMNetworkDesign:
        first_lstm = cast(FixedPointLSTMWithHardActivations, self.lstm)
        total_bits = first_lstm.fixed_point_config.total_bits
        frac_bits = first_lstm.fixed_point_config.frac_bits
        hidden_size = first_lstm.hidden_size
        input_size = first_lstm.input_size
        follow_up_linear_layers = cast(
            list[_FPLinear1dDesign],
            [cast(FPLinear1d, layer).translate() for layer in self.layers],
        )
        return LSTMNetworkDesign(
            lstm=first_lstm.translate(),
            linear_layers=follow_up_linear_layers,
            total_bits=total_bits,
            frac_bits=frac_bits,
            hidden_size=hidden_size,
            input_size=input_size,
        )

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = x[:, -1]
        return self.layers(x)


class FixedPointLSTMWithHardActivations(LSTM, Module):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        input_size: int,
        hidden_size: int,
        bias: bool,
    ) -> None:
        config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)

        class LayerFactory:
            def lstm(self, input_size: int, hidden_size: int, bias: bool) -> LSTMCell:
                return LSTMCell(
                    arithmetics=FixedPointArithmetics(config=config),
                    sigmoid_factory=HardSigmoid,
                    tanh_factory=HardTanh,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    bias=bias,
                )

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            batch_first=True,
            layers=LayerFactory(),
        )

        self._config = config

    @property
    def fixed_point_config(self) -> FixedPointConfig:
        return self._config

    def translate(self) -> Design:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._config.as_integer(value)

        cast_weights = lambda x: cast(list[list[list[int]]], x)
        cast_bias = lambda x: cast(list[list[int]], x)

        return FPLSTMCell(
            name="lstm_cell",
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            w_ih=cast_weights(float_to_signed_int(self.cell.linear_ih.weight.tolist())),
            w_hh=cast_weights(float_to_signed_int(self.cell.linear_hh.weight.tolist())),
            b_ih=cast_bias(float_to_signed_int(self.cell.linear_ih.bias.tolist())),
            b_hh=cast_bias(float_to_signed_int(self.cell.linear_hh.bias.tolist())),
            lower_bound_for_hard_sigmoid=cast(int, float_to_signed_int(-3)),
            upper_bound_for_hard_sigmoid=cast(int, float_to_signed_int(3)),
        )
