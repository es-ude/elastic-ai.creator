import math
import random

import numpy as np
import torch

from elasticai.creator.layers import QLSTMCell
from elasticai.creator.vhdl.generator.generator_functions import (
    float_array_to_hex_string,
    float_array_to_int,
)

from elasticai.creator.vhdl.generator.rom import Rom


def generate_rom_file(
    file_path: str,
    weights_or_bias_list: list[list[str]],
    nbits: int,
    name: str,
    index: int,
) -> None:
    """
    generates the rom files for the weights and bias
    Args:
        file_path (str): paths where files should be stored
        weights_or_bias_list (list[list[str]]): list with four lists with the hex strings for each weight or bias
        nbits (int): number of bits
        name (str): name for the file
        index (int): index where content is stored in weights_or_bias_list
    """
    with open(file_path, "w") as writer:
        weight_or_bias_array = weights_or_bias_list[index]
        addr_width = math.ceil(math.log2(len(weight_or_bias_array)))
        array_value = weight_or_bias_array

        rom = Rom(
            rom_name="rom_" + name,
            data_width=nbits,
            addr_width=addr_width,
            array_value=array_value,
        )
        rom_code = rom()
        for line in rom_code:
            writer.write(line + "\n")


def inference_model(
    lstm_signal_cell: QLSTMCell,
    frac_bits: int,
    nbits: int,
    input_size: int,
    hidden_size: int,
) -> tuple[list[str], list[str], np.array]:
    """
    do inference on defined QLSTM Cell
    Args:
        lstm_signal_cell (QLSTMCell): current QLSTM Cell
        frac_bits (int): number of fraction bits
        nbits (int): number of bits
        input_size (int): input size of QLSTM Cell
        hidden_size (int): hidden size of QLSTM Cell
    Returns:
        returns three lists/arrays
        the first and second list are the x_h input and cx of the lstm cell
        the third array is the hx of the lstm cell
    """
    torch.manual_seed(0)
    random.seed(0)

    input = torch.randn(2, 1, input_size)  # (time_steps, batch, input_size)
    hx = torch.randn(1, hidden_size)  # (batch, hidden_size), this is the hidden states
    cx = torch.randn(1, hidden_size)  # this the cell states
    # input = input/torch.max(abs(input))
    # hx = hx/torch.max(abs(hx))
    # cx = cx/torch.max(abs(cx))
    output = []
    for i in range(input.size()[0]):
        x_h_input = np.hstack(
            (input[i].detach().numpy().flatten(), hx.detach().numpy().flatten())
        )

        hx, cx = lstm_signal_cell(input[i], (hx, cx))
        output.append(hx)

        return (
            float_array_to_hex_string(
                x_h_input,
                frac_bits=frac_bits,
                nbits=nbits,
            ),
            float_array_to_hex_string(
                cx.detach().numpy().flatten(),
                frac_bits=frac_bits,
                nbits=nbits,
            ),
            float_array_to_int(hx.detach().numpy().flatten(), frac_bits=frac_bits),
        )


def define_weights_and_bias(
    lstm_signal_cell: QLSTMCell,
    frac_bits: int,
    nbits: int,
    len_weights: int,
    len_bias: int,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    calculates the weights and bias for the given QLSTM Cell
    Args:
        lstm_signal_cell (QLSTMCell): current QLSTM Cell
        frac_bits (int): number of fraction bits
        nbits (int): number of bits
        len_weights (int): (input_size + hidden_size) * hidden_size
        len_bias (int): hidden_size
    Returns:
        returns two lists, one for the weights and one for the bias
        in each list are four list of strings with the hex numbers of the weights or bias
    """
    for name, param in lstm_signal_cell.named_parameters():

        if name == "weight_ih":
            weight_ih = param.detach().numpy()
        elif name == "weight_hh":
            weight_hh = param.detach().numpy()
        elif name == "bias_ih":
            bias_ih = param.detach().numpy()
        elif name == "bias_hh":
            bias_hh = param.detach().numpy()

    weights = np.hstack((weight_ih, weight_hh)).flatten().flatten()
    bias = bias_hh + bias_ih

    wi = weights[len_weights * 0 : len_weights * 1]  # [Wii, Whi]
    wf = weights[len_weights * 1 : len_weights * 2]  # [Wif, Whf]
    wg = weights[len_weights * 2 : len_weights * 3]  # [Wig, Whg]
    wo = weights[len_weights * 3 : len_weights * 4]  # [Wio, Who]

    bi = bias[len_bias * 0 : len_bias * 1]  # B_ii+B_hi
    bf = bias[len_bias * 1 : len_bias * 2]  # B_if+B_hf
    bg = bias[len_bias * 2 : len_bias * 3]  # B_ig+B_hg
    bo = bias[len_bias * 3 : len_bias * 4]  # B_io+B_ho

    wi = float_array_to_hex_string(wi, frac_bits=frac_bits, nbits=nbits)
    wf = float_array_to_hex_string(wf, frac_bits=frac_bits, nbits=nbits)
    wg = float_array_to_hex_string(wg, frac_bits=frac_bits, nbits=nbits)
    wo = float_array_to_hex_string(wo, frac_bits=frac_bits, nbits=nbits)

    bi = float_array_to_hex_string(bi, frac_bits=frac_bits, nbits=nbits)
    bf = float_array_to_hex_string(bf, frac_bits=frac_bits, nbits=nbits)
    bg = float_array_to_hex_string(bg, frac_bits=frac_bits, nbits=nbits)
    bo = float_array_to_hex_string(bo, frac_bits=frac_bits, nbits=nbits)

    return [wi, wf, wg, wo], [bi, bf, bg, bo]
