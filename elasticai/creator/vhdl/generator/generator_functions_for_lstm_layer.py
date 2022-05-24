import random
from functools import partial

import numpy as np
import torch
from torch.nn import LSTM

from elasticai.creator.vhdl.number_representations import (
    FloatToSignedFixedPointConverter,
)


def float_list_to_fixed_point(values: list[float], frac_bits: int) -> list[int]:
    signed_fixed_point_converter = FloatToSignedFixedPointConverter(
        bits_used_for_fraction=frac_bits, strict=False
    )
    return list(map(signed_fixed_point_converter, values))


def inference_lstm_layer(
    lstm_layer: LSTM,
    frac_bits: int,
    input_size: int,
    hidden_size: int,
) -> tuple[list[int], list[int], np.array]:
    """
    do inference on the given multilayer LSTM
    Args:
        lstm_layer (QLSTMCell): current LSTM layer
        frac_bits (int): number of fraction bits
        input_size (int): input size of LSTM layer
        hidden_size (int): hidden size of LSTM layer
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

    for i in range(input.size()[0]):
        x_h_input = np.hstack(
            (input[i].detach().numpy().flatten(), hx.detach().numpy().flatten())
        )
        outputs, (hx, cx) = lstm_layer(input[i], (hx, cx))
        c_array = cx.detach().numpy().flatten()
        h_out_array = hx.detach().numpy().flatten()
        return (
            float_list_to_fixed_point(
                x_h_input,
                frac_bits=frac_bits,
            ),
            float_list_to_fixed_point(
                c_array,
                frac_bits=frac_bits,
            ),
            float_list_to_fixed_point(h_out_array, frac_bits=frac_bits),
        )


def define_weights_and_bias_of_lstm_layer(
    lstm_layer: LSTM,
    frac_bits: int,
    len_weights: int,
    len_bias: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    calculates the weights and bias for the 1st layer of the given multilayer LSTM
    Args:
        lstm_layer (torch.nn.LSTM): current LSTM
        frac_bits (int): number of fraction bits
        len_weights (int): (input_size + hidden_size) * hidden_size
        len_bias (int): hidden_size
    Returns:
        returns two lists, one for the weights and one for the bias
        in each list are four list of strings with the hex numbers of the weights or bias
    """
    for name, param in lstm_layer.named_parameters():
        if name == "weight_ih_l0":
            weight_ih = param.detach().numpy()
        elif name == "weight_hh_l0":
            weight_hh = param.detach().numpy()
        elif name == "bias_ih_l0":
            bias_ih = param.detach().numpy()
        elif name == "bias_hh_l0":
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

    to_fixed_point = partial(float_list_to_fixed_point, frac_bits=frac_bits)
    fixed_point_weights = list(map(to_fixed_point, [wi, wf, wg, wo]))
    fixed_point_bias = list(map(to_fixed_point, [bi, bf, bg, bo]))

    return fixed_point_weights, fixed_point_bias
