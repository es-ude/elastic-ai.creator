import torch
import random
import numpy as np
import math
from elasticai.creator.layers import QLSTMCell
from elasticai.creator.vhdl.generator.generator_functions import get_file_path_string
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl
from elasticai.creator.vhdl.generator.lstm_testbench_generator import LSTMCellTestBench
from elasticai.creator.vhdl.generator.rom import Rom
from elasticai.creator.vhdl.generator.precomputed_scalar_function import Sigmoid, Tanh

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


def format_array_to_string_without_prefix(arr, nbits=16):
    string_to_return = ""

    for i in range(2 ** math.ceil(math.log2(len(arr)))):
        if i < len(arr):
            string_to_return += 'x"' + int_to_hex(arr[i], nbits)[2:] + '",'
        else:
            string_to_return += 'x"' + int_to_hex(0, nbits)[2:] + '",'
    string_to_return = string_to_return[:-1]

    return string_to_return


def float_array_to_int(float_array, frac_bits=8):
    scaled_array = float_array * 2 ** frac_bits
    int_array = scaled_array.astype(np.int16)
    return int_array


def float_array_to_string(float_array, vhdl_prefix=None, frac_bits=8, nbits=16):
    scaled_array = float_array * 2 ** frac_bits
    int_array = scaled_array.astype(np.int16)
    return format_array_to_string(int_array, vhdl_prefix, nbits)


def float_array_to_string_without_prefix(float_array, frac_bits=8, nbits=16):
    scaled_array = float_array * 2 ** frac_bits
    int_array = scaled_array.astype(np.int16)
    return format_array_to_string_without_prefix(int_array, nbits)


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


def generate_rom_file(
    file_path,
    weights_or_bias_list: list,
    frac_bits: int,
    nbits: int,
    name: str,
    index: int,
):
    with open(file_path, "w") as writer:
        weight_or_bias_array = weights_or_bias_list[index]
        addr_width = math.ceil(math.log2(len(weight_or_bias_array)))
        array_value = float_array_to_string_without_prefix(
            float_array=weight_or_bias_array, frac_bits=frac_bits, nbits=nbits
        )
        rom = Rom(
            rom_name="rom_" + name,
            data_width=nbits,
            addr_width=addr_width,
            array_value=array_value,
        )
        rom_code = rom()
        for line in rom_code:
            writer.write(line + "\n")


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

    ### generate source files for use-case ###

    ## generate weights source files ##
    file_path_wi = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/systemTests",
        file_name="wi_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_wi,
        weights_or_bias_list=weights_list,
        frac_bits=frac_bits,
        nbits=nbits,
        name="wi",
        index=0,
    )
    file_path_wf = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/systemTests",
        file_name="wf_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_wf,
        weights_or_bias_list=weights_list,
        frac_bits=frac_bits,
        nbits=nbits,
        name="wf",
        index=1,
    )
    file_path_wg = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/systemTests",
        file_name="wg_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_wg,
        weights_or_bias_list=weights_list,
        frac_bits=frac_bits,
        nbits=nbits,
        name="wg",
        index=2,
    )
    file_path_wo = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/systemTests",
        file_name="wo_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_wo,
        weights_or_bias_list=weights_list,
        frac_bits=frac_bits,
        nbits=nbits,
        name="wo",
        index=3,
    )

    ## generate bias source files ##
    file_path_bi = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/systemTests",
        file_name="bi_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_bi,
        weights_or_bias_list=bias_list,
        frac_bits=frac_bits,
        nbits=nbits,
        name="bi",
        index=0,
    )
    file_path_bf = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/systemTests",
        file_name="bf_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_bf,
        weights_or_bias_list=bias_list,
        frac_bits=frac_bits,
        nbits=nbits,
        name="bf",
        index=1,
    )
    file_path_bg = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/systemTests",
        file_name="bg_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_bg,
        weights_or_bias_list=bias_list,
        frac_bits=frac_bits,
        nbits=nbits,
        name="bg",
        index=2,
    )
    file_path_bo = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/systemTests",
        file_name="bo_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_bo,
        weights_or_bias_list=bias_list,
        frac_bits=frac_bits,
        nbits=nbits,
        name="bo",
        index=3,
    )

    ## generate sigmoid and tanh activation source files ##
    file_path_sigmoid = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/systemTests",
        file_name="sigmoid.vhd",
    )

    with open(file_path_sigmoid, "w") as writer:
        sigmoid = Sigmoid(
            data_width=nbits, frac_width=frac_bits, x=np.linspace(-2.5, 2.5, 256)
        )
        sigmoid_code = sigmoid()
        for line in sigmoid_code:
            writer.write(line + "\n")

    file_path_tanh = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/systemTests",
        file_name="tanh.vhd",
    )

    with open(file_path_tanh, "w") as writer:
        tanh = Tanh(data_width=nbits, frac_width=frac_bits, x=np.linspace(-1, 1, 256))
        tanh_code = tanh()
        for line in tanh_code:
            writer.write(line + "\n")

    ### generate testbench file for use-case ###
    file_path_testbench = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/systemTests",
        file_name="lstm_cell_tb.vhd",
    )

    with open(file_path_testbench, "w") as writer:
        lstm_cell = LSTMCellTestBench(
            data_width=nbits,
            frac_width=frac_bits,
            input_size=input_size,
            hidden_size=hidden_size,
            test_x_h_data=float_array_to_string_without_prefix(
                x_h_test_input,
                frac_bits=frac_bits,
                nbits=nbits,
            ),
            test_c_data=float_array_to_string_without_prefix(
                c_test_input,
                frac_bits=frac_bits,
                nbits=nbits,
            ),
            h_out=list(h_output),
            component_name="lstm_cell",
        )
        lstm_cell_code = lstm_cell()
        for line in lstm_cell_code:
            writer.write(line + "\n")

    # indent all lines of the files
    format_vhdl(file_path=file_path_wi)
    format_vhdl(file_path=file_path_wf)
    format_vhdl(file_path=file_path_wg)
    format_vhdl(file_path=file_path_wo)
    format_vhdl(file_path=file_path_bi)
    format_vhdl(file_path=file_path_bf)
    format_vhdl(file_path=file_path_bg)
    format_vhdl(file_path=file_path_bo)
    format_vhdl(file_path=file_path_sigmoid)
    format_vhdl(file_path=file_path_tanh)
    format_vhdl(file_path=file_path_testbench)
