from functools import partial

import numpy as np
import torch
from torch.nn import LSTM, LSTMCell

from elasticai.creator.vhdl.components.rom_component import RomComponent
from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    float_values_to_fixed_point,
)


def generate_rom_file(
    file_path: str,
    rom_name: str,
    values: list[FixedPoint],
    resource_option: str,
) -> None:
    """
    generates the rom files for a given list of values
    Args:
        file_path (str): paths where files should be stored
        values (list[list[int]]): list with four lists with the fixed point values for each weight or bias
        rom_name (str): name for the file
        resource_option (str): resource option
    """
    with open(file_path, "w") as writer:
        rom = RomComponent(
            rom_name=rom_name,
            values=values,
            resource_option=resource_option,
        )
        rom_code = rom()
        for line in rom_code:
            writer.write(line + "\n")


def inference_model_on_random_data(
    lstm: LSTMCell | LSTM,
    total_bits: int,
    frac_bits: int,
) -> tuple[list[FixedPoint], list[FixedPoint], list[FixedPoint]]:
    """
    do inference on defined LSTM Cell or LSTM
    Args:
        lstm (LSTMCell | LSTM): current LSTM Cell or LSTM
        total_bits (int): total number of bits for one fixed point number
        frac_bits (int): number of fraction bits for one fixed point number
    Returns:
        returns three lists/arrays
        the first and second list are the x_h input_data and cx of the lstm cell
        the third array is the hx of the lstm cell
    """
    torch.manual_seed(0)

    input_data = torch.randn(2, 1, lstm.input_size)  # (time_steps, batch, input_size)
    hx = torch.randn(1, lstm.hidden_size)  # this is the hidden states
    cx = torch.randn(1, lstm.hidden_size)  # this the cell states

    x_h_input = np.hstack(([], [], []))

    for i in range(input_data.size()[0]):
        x_h_input = np.hstack(
            (input_data[i].detach().numpy().flatten(), hx.detach().numpy().flatten())
        )
        result = lstm(input_data[i], (hx, cx))

        if isinstance(lstm, LSTMCell):
            hx, cx = result
        else:
            _, (hx, cx) = result

    to_fp = partial(
        float_values_to_fixed_point, total_bits=total_bits, frac_bits=frac_bits
    )

    return (
        to_fp(x_h_input),
        to_fp(cx.detach().numpy().flatten()),
        to_fp(hx.detach().numpy().flatten()),
    )


def extract_fixed_point_weights_and_bias(
    lstm: LSTMCell | LSTM, total_bits: int, frac_bits: int
) -> tuple[list[list[FixedPoint]], list[list[FixedPoint]]]:
    """
    calculates the weights and bias for the 1st layer of the given multilayer LSTM
    or the given LSTMCell
    Args:
        lstm (LSTMCell | LSTM): current LSTM object
        total_bits (int): total number of bits for one fixed point number
        frac_bits (int): number of fraction bits for one fixed point number
    Returns:
        returns two lists, one for the weights and one for the bias
        in each list are four list of FixedPoint numbers of the weights or bias
    """
    if isinstance(lstm, LSTMCell):
        weight_ih = lstm.weight_ih.detach().numpy()
        weight_hh = lstm.weight_hh.detach().numpy()
        bias_ih = lstm.bias_ih.detach().numpy()
        bias_hh = lstm.bias_hh.detach().numpy()
    else:
        weight_ih = lstm.weight_ih_l0.detach().numpy()
        weight_hh = lstm.weight_hh_l0.detach().numpy()
        bias_ih = lstm.bias_ih_l0.detach().numpy()
        bias_hh = lstm.bias_hh_l0.detach().numpy()

    weights = np.hstack((weight_ih, weight_hh)).flatten().flatten()
    bias = bias_hh + bias_ih

    len_weights = (lstm.input_size + lstm.hidden_size) * lstm.hidden_size
    len_bias = lstm.hidden_size

    wi = weights[len_weights * 0 : len_weights * 1]  # [Wii, Whi]
    wf = weights[len_weights * 1 : len_weights * 2]  # [Wif, Whf]
    wg = weights[len_weights * 2 : len_weights * 3]  # [Wig, Whg]
    wo = weights[len_weights * 3 : len_weights * 4]  # [Wio, Who]

    bi = bias[len_bias * 0 : len_bias * 1]  # B_ii+B_hi
    bf = bias[len_bias * 1 : len_bias * 2]  # B_if+B_hf
    bg = bias[len_bias * 2 : len_bias * 3]  # B_ig+B_hg
    bo = bias[len_bias * 3 : len_bias * 4]  # B_io+B_ho

    to_fp = partial(
        float_values_to_fixed_point, total_bits=total_bits, frac_bits=frac_bits
    )

    fixed_point_weights = list(map(to_fp, [wi, wf, wg, wo]))
    fixed_point_bias = list(map(to_fp, [bi, bf, bg, bo]))

    return fixed_point_weights, fixed_point_bias
