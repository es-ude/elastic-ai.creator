import random
from itertools import filterfalse
from typing import Dict, Iterable

import torch
from torch import nn as nn

from elasticai.creator.layers import QLSTMCell
from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import FloatToSignedFixedPointConverter, \
    FloatToHexFixedPointStringConverter


def _elastic_ai_creator_lstm() -> QLSTMCell:
    return QLSTMCell(
        1, 1, state_quantizer=nn.Identity(), weight_quantizer=nn.Identity()
    )


def _ensure_reproducibility():
    torch.manual_seed(0)
    random.seed(0)


def generate_signal_definitions_for_lstm(data_width: int, frac_width: int) -> Dict:
    """
    returns Dict of signals names as key and their definition as value
    """
    dict_of_signals = {}
    _ensure_reproducibility()
    lstm_single_cell = _elastic_ai_creator_lstm()

    # weight_ih_l[k] : `(W_ii|W_if|W_ig|W_io)`
    # weight_hh_l[k] : `(W_hi|W_hf|W_hg|W_ho)`
    # bias_ih_l[k] :  `(b_ii|b_if|b_ig|b_io)`
    # bias_hh_l[k] :  `(W_hi|W_hf|W_hg|b_ho)`

    b = [0, 0, 0, 0]

    floats_to_signed_fixed_point_converter = FloatToSignedFixedPointConverter(
        bits_used_for_fraction=frac_width, strict=False
    )
    float_to_hex_fixed_point_string_converter = FloatToHexFixedPointStringConverter(
        total_bit_width=data_width,
        as_signed_fixed_point=floats_to_signed_fixed_point_converter,
    )

    def update_signals(signal_names: Iterable[str], signal_values: Iterable[float]):
        print(signal_values)
        dict_of_signals.update(
            (
                (
                    signal_name,
                    f'X"{float_to_hex_fixed_point_string_converter(signal_value)}"',
                )
                for signal_name, signal_value in zip(signal_names, signal_values)
            )
        )

    for name, param in lstm_single_cell.named_parameters():
        if name == "weight_ih":
            signal_names = ("wii", "wif", "wig", "wio")
            update_signals(signal_names, param)
        elif name == "weight_hh":
            update_signals(
                signal_names=("whi", "whf", "whg", "who"), signal_values=param
            )
        elif name in ("bias_ih", "bias_hh"):
            b = [new_value + old_value for old_value, new_value in zip(b, param)]
        else:
            dict_of_signals.update("should not come to here.")

    update_signals(signal_names=("bi", "bf", "bg", "bo"), signal_values=b)
    return dict_of_signals


class LstmCell:
    """
    ```
    cell = LstmCell(component_name="my_comp", data_width=16, frac_width=8)
    with open("file.vhd", "w") as f:
        for line in cell():
            f.writeline(line)
    ```
    """
    def __init__(self, component_name, data_width, frac_width):
        self.component_name = component_name
        self.data_width = data_width
        self.frac_width = frac_width

    def __call__(self) -> Code:
        template = read_text("elasticai.creator.vhdl.generator.templates", "lstm_cell.tpl.vhd")
        signal_defs = generate_signal_definitions_for_lstm(data_width=self.data_width, frac_width=self.frac_width)
        parameters = signal_defs | {'data_width': self.data_width, 'frac_width': self.frac_width}
        code = template.format(**parameters)

        def line_is_empty(line):
            return len(line) == 0
        yield from filterfalse(line_is_empty, map(str.strip, code.splitlines()))
