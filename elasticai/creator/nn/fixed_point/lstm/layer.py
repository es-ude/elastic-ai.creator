from collections import OrderedDict
from typing import cast

import torch

from elasticai.creator.base_modules.lstm import LSTM
from elasticai.creator.base_modules.lstm_cell import LSTMCell
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.nn.fixed_point.hard_sigmoid import HardSigmoid
from elasticai.creator.nn.fixed_point.hard_tanh import HardTanh
from elasticai.creator.nn.fixed_point.linear import Linear
from elasticai.creator.nn.fixed_point.linear.design import Linear as _LinearDesign
from elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell import FPLSTMCell
from elasticai.creator.nn.fixed_point.lstm.design.lstm import LSTMNetworkDesign
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design_creator import DesignCreator

from .._math_operations import MathOperations


class LSTMNetwork(DesignCreator, torch.nn.Module):
    def __init__(self, layers: list[torch.nn.Module]):
        super().__init__()
        self.lstm = layers[0]
        self.layer_names = [f"fp_linear_1d_{i}" for i in range(len(layers[1:]))]
        self.layers = torch.nn.Sequential(
            OrderedDict(
                {name: layer for name, layer in zip(self.layer_names, layers[1:])}
            )
        )

    def create_design(self, name: str) -> LSTMNetworkDesign:
        first_lstm = cast(FixedPointLSTMWithHardActivations, self.lstm)
        total_bits = first_lstm.fixed_point_config.total_bits
        frac_bits = first_lstm.fixed_point_config.frac_bits
        hidden_size = first_lstm.hidden_size
        input_size = first_lstm.input_size
        follow_up_linear_layers = cast(
            list[_LinearDesign],
            [
                cast(Linear, layer).create_design(self.layer_names[i])
                for i, layer in enumerate(self.layers)
            ],
        )
        return LSTMNetworkDesign(
            lstm=first_lstm.create_design(),
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


class FixedPointLSTMWithHardActivations(LSTM, DesignCreator):
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
                def activation(constructor):
                    def wrapped_constructor():
                        return constructor(total_bits=total_bits, frac_bits=frac_bits)

                    return wrapped_constructor

                return LSTMCell(
                    operations=MathOperations(config=config),
                    sigmoid_factory=activation(HardSigmoid),
                    tanh_factory=activation(HardTanh),
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

    def create_design(self, name: str = "lstm_cell") -> Design:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._config.as_integer(value)

        cast_weights = lambda x: cast(list[list[list[int]]], x)
        cast_bias = lambda x: cast(list[list[int]], x)

        return FPLSTMCell(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            w_ih=cast_weights(float_to_signed_int(self.cell.linear_ih.weight.tolist())),
            w_hh=cast_weights(float_to_signed_int(self.cell.linear_hh.weight.tolist())),
            b_ih=cast_bias(float_to_signed_int(self.cell.linear_ih.bias.tolist())),
            b_hh=cast_bias(float_to_signed_int(self.cell.linear_hh.bias.tolist())),
        )
