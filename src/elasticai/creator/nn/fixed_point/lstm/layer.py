from collections import OrderedDict
from typing import cast

import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.base_modules.lstm import LSTM
from elasticai.creator.base_modules.lstm_cell import LSTMCell
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.hard_sigmoid import HardSigmoid
from elasticai.creator.nn.fixed_point.hard_tanh import HardTanh
from elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell import FPLSTMCell
from elasticai.creator.nn.fixed_point.lstm.design.lstm import LSTMNetworkDesign
from elasticai.creator.vhdl.design.design import Design

from ..math_operations import MathOperations
from .design.testbench import LSTMTestBench


class LSTMNetwork(DesignCreatorModule):
    def __init__(self, layers: list[torch.nn.Module]):
        super().__init__()
        self.lstm = layers[0]
        self.layer_names = [f"fp_linear_{i}" for i in range(len(layers[1:]))]
        if len(self.layer_names) > 1:
            raise NotImplementedError
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

        return LSTMNetworkDesign(
            lstm=first_lstm.create_design(),
            linear_layer=self.layers[0].create_design(self.layer_names[0]),
            total_bits=total_bits,
            frac_bits=frac_bits,
            hidden_size=hidden_size,
            input_size=input_size,
        )

    def create_testbench(self, test_bench_name, uut: Design) -> LSTMTestBench:
        return LSTMTestBench(test_bench_name, uut)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = x[:, -1]
        return self.layers(x)


class FixedPointLSTMWithHardActivations(DesignCreatorModule, LSTM):
    """
    Use only together with the above `LSTMNetwork`.
    There is no single hw design corresponding to this sw layer.
    Instead, the design of the `LSTMNetwork` handles most of the tasks,
    that are performed by `FixedPointLSTMWithHardActivations`
    """

    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        input_size: int,
        hidden_size: int,
        bias: bool,
    ) -> None:
        params = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
        config = FxpArithmetic(params)

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
    def fixed_point_config(self) -> FxpArithmetic:
        return self._config

    def create_design(self, name: str = "lstm_cell") -> Design:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._config.cut_as_integer(value)

        def cast_weights(x):
            return cast(list[list[list[int]]], x)

        def cast_bias(x):
            return cast(list[list[int]], x)

        return FPLSTMCell(
            name=name,
            hardtanh=self.cell.tanh.create_design(f"{name}_hardtanh"),
            hardsigmoid=self.cell.sigmoid.create_design(f"{name}_hardsigmoid"),
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            w_ih=cast_weights(float_to_signed_int(self.cell.linear_ih.weight.tolist())),
            w_hh=cast_weights(float_to_signed_int(self.cell.linear_hh.weight.tolist())),
            b_ih=cast_bias(float_to_signed_int(self.cell.linear_ih.bias.tolist())),
            b_hh=cast_bias(float_to_signed_int(self.cell.linear_hh.bias.tolist())),
        )
