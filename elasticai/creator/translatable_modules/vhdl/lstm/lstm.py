from collections import OrderedDict
from typing import cast

import torch.nn

from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.nn.lstm import FixedPointLSTMWithHardActivations as _nnLSTM
from elasticai.creator.translatable_modules.vhdl.lstm.design import LSTMNetworkDesign
from elasticai.creator.translatable_modules.vhdl.lstm.fp_lstm_cell import FPLSTMCell
from elasticai.creator.translatable_modules.vhdl.module import Module


class LSTMNetwork(torch.nn.Module):
    def __init__(self, layers: list[torch.nn.Module]):
        super().__init__()
        self.layers = torch.nn.Sequential(
            OrderedDict({f"{i}": layer for i, layer in enumerate(layers)})
        )

    def translate(self) -> LSTMNetworkDesign:
        children = list(self.layers.children())
        first_lstm = cast(FixedPointLSTMWithHardActivations, children[0])
        total_bits = first_lstm.fixed_point_config.total_bits
        frac_bits = first_lstm.fixed_point_config.frac_bits
        hidden_size = first_lstm.hidden_size
        input_size = first_lstm.input_size
        return LSTMNetworkDesign(
            first_lstm.translate(),
            total_bits=total_bits,
            frac_bits=frac_bits,
            hidden_size=hidden_size,
            input_size=input_size,
        )

    def forward(self, x):
        return self.layers(x)


class FixedPointLSTMWithHardActivations(_nnLSTM, Module):
    def translate(self) -> Design:
        def to_list(tensor: torch.Tensor) -> list:
            return tensor.detach().numpy().tolist()

        return FPLSTMCell(
            total_bits=self.fixed_point_config.total_bits,
            frac_bits=self.fixed_point_config.frac_bits,
            w_ih=to_list(self.cell.linear_ih.weight),
            w_hh=to_list(self.cell.linear_hh.weight),
            b_ih=to_list(self.cell.linear_ih.bias),
            b_hh=to_list(self.cell.linear_hh.bias),
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
