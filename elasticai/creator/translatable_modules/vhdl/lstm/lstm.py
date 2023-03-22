from collections import OrderedDict
from typing import cast

import torch

from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.nn.lstm import FixedPointLSTMWithHardActivations as _nnLSTM
from elasticai.creator.translatable_modules.vhdl.fp_linear_1d import FPLinear1d
from elasticai.creator.translatable_modules.vhdl.fp_linear_1d.design import (
    FPLinear1d as _FPLinear1dDesign,
)
from elasticai.creator.translatable_modules.vhdl.lstm.design import LSTMNetworkDesign
from elasticai.creator.translatable_modules.vhdl.lstm.fp_lstm_cell import FPLSTMCell
from elasticai.creator.translatable_modules.vhdl.module import Module


class LSTMNetwork(torch.nn.Module):
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


class FixedPointLSTMWithHardActivations(_nnLSTM, Module):
    def translate(self) -> Design:
        def to_list(tensor: torch.Tensor) -> list[float]:
            return tensor.detach().numpy().tolist()

        def convert_float_to_signed_integer(
            value: float | list[float] | list[list[float]] | list[list[list[float]]],
        ):
            if isinstance(value, list):
                return list(map(convert_float_to_signed_integer, value))
            return self.fixed_point_config.as_integer(value)

        def convert_tensor(values: torch.Tensor) -> list[int]:
            return list(map(convert_float_to_signed_integer, to_list(values)))

        return FPLSTMCell(
            name="lstm_cell",
            total_bits=self.fixed_point_config.total_bits,
            frac_bits=self.fixed_point_config.frac_bits,
            w_ih=convert_tensor(self.cell.linear_ih.weight),
            w_hh=convert_tensor(self.cell.linear_hh.weight),
            b_ih=convert_tensor(self.cell.linear_ih.bias),
            b_hh=convert_tensor(self.cell.linear_hh.bias),
            lower_bound_for_hard_sigmoid=convert_float_to_signed_integer(-3),
            upper_bound_for_hard_sigmoid=convert_float_to_signed_integer(3),
        )

    def __init__(
        self, total_bits: int, frac_bits: int, input_size: int, hidden_size: int
    ):
        super().__init__(
            total_bits=total_bits,
            frac_bits=frac_bits,
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            batch_first=True,
        )
