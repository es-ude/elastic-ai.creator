import torch
import random
import numpy as np
import math
from elasticai.creator.layers import QLSTMCell

"""
this module is from the lstm repo from Chao
"""


def int_to_hex(val, nbits):
    if val < 0:
        return hex((val + (1 << nbits)) % (1 << nbits))
    else:
        return "{0:#0{1}x}".format(val, 2 + int(nbits / 4))


def fixed_point_multiply(x, y, frac_bits=8):
    return int(x * y / (2 ** frac_bits))


def format_array_to_string(arr, vhdl_prefix=None, nbits=16):
    if vhdl_prefix is None:
        string_to_return = "X_MEM :="
    else:
        string_to_return = vhdl_prefix
    string_to_return += " ("

    for i in range(2 ** math.ceil(math.log2(len(arr)))):
        if i < len(arr):
            string_to_return += 'x"' + int_to_hex(arr[i], nbits)[2:] + '",'
        else:
            string_to_return += 'x"' + int_to_hex(0, nbits)[2:] + '",'
    string_to_return = string_to_return[:-1] + ");"

    return string_to_return


def float_array_to_int(float_array, frac_bits=8):
    scaled_array = float_array * 2 ** frac_bits
    int_array = scaled_array.astype(np.int16)
    return int_array


def float_array_to_string(float_array, vhdl_prefix=None, frac_bits=8, nbits=16):
    scaled_array = float_array * 2 ** frac_bits
    int_array = scaled_array.astype(np.int16)
    return format_array_to_string(int_array, vhdl_prefix, nbits)


def int_to_bit(val, nbits):
    format_str = "{0:0" + str(nbits) + "b}"
    if val < 0:
        return bin((val + (1 << nbits)) % (1 << nbits))[2:]
    else:
        return format_str.format(val)


def mem_array_to_dat_file(file_name="", float_arr=None, nbits=16, frac_bits=8):
    scaled_array = float_arr * 2 ** frac_bits
    int_array = scaled_array.astype(np.int16)
    # file open
    if file_name == "":
        print("you must specify the file name where to dump your array.\r\n")
        return

    string_to_write = ""
    # if len(int_array) < 2**math.ceil(math.log2(len(int_array))):
    for i in range(2 ** math.ceil(math.log2(len(int_array)))):
        if i < len(int_array):
            string_to_write += int_to_bit(int_array[i], nbits) + "\r"
        else:
            string_to_write += int_to_bit(0, nbits) + "\r"

    txt_file = open(file_name, "w")
    txt_file.write(string_to_write[:-1])
    txt_file.close()


def define_lstm_cell(input_size, hidden_size) -> QLSTMCell:
    return QLSTMCell(
        input_size=input_size,
        hidden_size=hidden_size,
        state_quantizer=lambda x: x,
        weight_quantizer=lambda x: x,
    )


def define_weights_and_bias(lstm_signal_cell, frac_bits, nbits, len_weights, len_bias):
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

    Wi = weights[len_weights * 0 : len_weights * 1]  # [Wii, Whi]
    Wf = weights[len_weights * 1 : len_weights * 2]  # [Wif, Whf]
    Wg = weights[len_weights * 2 : len_weights * 3]  # [Wig, Whg]
    Wo = weights[len_weights * 3 : len_weights * 4]  # [Wio, Who]

    Bi = bias[len_bias * 0 : len_bias * 1]  # B_ii+B_hi
    Bf = bias[len_bias * 1 : len_bias * 2]  # B_if+B_hf
    Bg = bias[len_bias * 2 : len_bias * 3]  # B_ig+B_hg
    Bo = bias[len_bias * 3 : len_bias * 4]  # B_io+B_ho

    print(
        float_array_to_string(
            float_array=Wi,
            vhdl_prefix="signal ROM : WI_ARRAY :=",
            frac_bits=frac_bits,
            nbits=nbits,
        )
    )
    print(
        float_array_to_string(
            float_array=Wf,
            vhdl_prefix="signal ROM : WF_ARRAY :=",
            frac_bits=frac_bits,
            nbits=nbits,
        )
    )
    print(
        float_array_to_string(
            float_array=Wg,
            vhdl_prefix="signal ROM : WG_ARRAY :=",
            frac_bits=frac_bits,
            nbits=nbits,
        )
    )
    print(
        float_array_to_string(
            float_array=Wo,
            vhdl_prefix="signal ROM : WO_ARRAY :=",
            frac_bits=frac_bits,
            nbits=nbits,
        )
    )

    print(
        float_array_to_string(
            float_array=Bi,
            vhdl_prefix="signal ROM : BI_ARRAY:=",
            frac_bits=frac_bits,
            nbits=nbits,
        )
    )
    print(
        float_array_to_string(
            float_array=Bf,
            vhdl_prefix="signal ROM : BF_ARRAY:=",
            frac_bits=frac_bits,
            nbits=nbits,
        )
    )
    print(
        float_array_to_string(
            float_array=Bg,
            vhdl_prefix="signal ROM : BG_ARRAY:=",
            frac_bits=frac_bits,
            nbits=nbits,
        )
    )
    print(
        float_array_to_string(
            float_array=Bo,
            vhdl_prefix="signal ROM : BO_ARRAY:=",
            frac_bits=frac_bits,
            nbits=nbits,
        )
    )
    return [Wi, Wf, Wg, Wo], [Bi, Bf, Bg, Bo]


def inference_model(lstm_signal_cell, frac_bits, nbits, input_size, hidden_size):
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
        print("===========round%s===========" % str(i))
        # print("input_x",input[i])
        # print("h_x",hx)
        # print("c_x",cx)
        x_h_input = np.hstack(
            (input[i].detach().numpy().flatten(), hx.detach().numpy().flatten())
        )
        print(
            float_array_to_string(
                x_h_input,
                vhdl_prefix="signal test_x_h_data : X_H_ARRAY := ",
                frac_bits=frac_bits,
                nbits=nbits,
            )
        )
        print(
            float_array_to_string(
                cx.detach().numpy().flatten(),
                vhdl_prefix="signal test_c_data : C_ARRAY := ",
                frac_bits=frac_bits,
                nbits=nbits,
            )
        )

        hx, cx = lstm_signal_cell(input[i], (hx, cx))
        # print("h_x out:", hx)
        # print("c_x out:", cx)
        print("h_out:")
        print(float_array_to_int(hx.detach().numpy().flatten(), frac_bits=frac_bits))
        output.append(hx)

        return (
            x_h_input,
            cx.detach().numpy().flatten(),
            float_array_to_int(hx.detach().numpy().flatten(), frac_bits=frac_bits),
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    frac_bits = 8
    nbits = 16
    input_size = 5
    hidden_size = 20
    len_weights = (input_size + hidden_size) * hidden_size
    len_bias = hidden_size

    lstm_cell = define_lstm_cell(input_size, hidden_size)
    weights_list, bias_list = define_weights_and_bias(
        lstm_cell, frac_bits, nbits, len_weights, len_bias
    )
    print("weights_list", weights_list)
    print("bias_list", bias_list)
    x_h_test_input, c_test_input, h_output = inference_model(
        lstm_cell, frac_bits, nbits, input_size, hidden_size
    )
    print("x_h_test_input", x_h_test_input)
    print("c_test_input", c_test_input)
    print("h_output", h_output)
