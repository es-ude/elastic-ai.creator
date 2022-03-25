import sys
import os
from argparse import ArgumentParser
import shutil
from paths import ROOT_DIR
import torch
import random
import numpy as np
from nptyping import NDArray
import math
from elasticai.creator.layers import QLSTMCell
from elasticai.creator.vhdl.generator.generator_functions import (
    get_file_path_string,
    float_array_to_int,
    float_array_to_hex_string,
)
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl
from elasticai.creator.vhdl.generator.lstm_testbench_generator import LSTMCellTestBench
from elasticai.creator.vhdl.generator.rom import Rom
from elasticai.creator.vhdl.generator.precomputed_scalar_function import Sigmoid, Tanh
from elasticai.creator.vhdl.language import form_to_hex_list

"""
this module generates all vhd files for a single lstm cell
"""


def define_lstm_cell(input_size: int, hidden_size: int) -> QLSTMCell:
    """
    returns a QLSTM Cell with the given input and hidden size
    Args:
        input_size (int): input size of QLSTM Cell
        hidden_size (int): hidden size of QLSTM Cell
    Returns:
        returns the corresponding QLSTM Cell
    """
    return QLSTMCell(
        input_size=input_size,
        hidden_size=hidden_size,
        state_quantizer=lambda x: x,
        weight_quantizer=lambda x: x,
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


def inference_model(
    lstm_signal_cell: QLSTMCell,
    frac_bits: int,
    nbits: int,
    input_size: int,
    hidden_size: int,
) -> tuple[list[str], list[str], NDArray[int]]:
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


if __name__ == "__main__":
    args = sys.argv[1:]
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--path",
        help="relative path from project root to folder for generated vhd files",
        required=True,
    )
    args = arg_parser.parse_args(args)
    if not os.path.isdir(ROOT_DIR + "/" + args.path):
        os.mkdir(ROOT_DIR + "/" + args.path)

    torch.manual_seed(0)
    random.seed(0)
    current_frac_bits = 8
    current_nbits = 16
    current_input_size = 5
    current_hidden_size = 20
    current_len_weights = (
        current_input_size + current_hidden_size
    ) * current_hidden_size
    current_len_bias = current_hidden_size

    lstm_cell = define_lstm_cell(current_input_size, current_hidden_size)
    weights_list, bias_list = define_weights_and_bias(
        lstm_cell,
        current_frac_bits,
        current_nbits,
        current_len_weights,
        current_len_bias,
    )
    print("weights_list", weights_list)
    print("bias_list", bias_list)
    x_h_test_input, c_test_input, h_output = inference_model(
        lstm_cell,
        current_frac_bits,
        current_nbits,
        current_input_size,
        current_hidden_size,
    )
    print("x_h_test_input", x_h_test_input)
    print("c_test_input", c_test_input)
    print("h_output", h_output)

    ### generate source files for use-case ###

    ## generate weights source files ##
    file_path_wi = get_file_path_string(
        relative_path_from_project_root=args.path,
        file_name="wi_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_wi,
        weights_or_bias_list=weights_list,
        nbits=current_nbits,
        name="wi",
        index=0,
    )
    file_path_wf = get_file_path_string(
        relative_path_from_project_root=args.path,
        file_name="wf_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_wf,
        weights_or_bias_list=weights_list,
        nbits=current_nbits,
        name="wf",
        index=1,
    )
    file_path_wg = get_file_path_string(
        relative_path_from_project_root=args.path,
        file_name="wg_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_wg,
        weights_or_bias_list=weights_list,
        nbits=current_nbits,
        name="wg",
        index=2,
    )
    file_path_wo = get_file_path_string(
        relative_path_from_project_root=args.path,
        file_name="wo_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_wo,
        weights_or_bias_list=weights_list,
        nbits=current_nbits,
        name="wo",
        index=3,
    )

    ## generate bias source files ##
    file_path_bi = get_file_path_string(
        relative_path_from_project_root=args.path,
        file_name="bi_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_bi,
        weights_or_bias_list=bias_list,
        nbits=current_nbits,
        name="bi",
        index=0,
    )
    file_path_bf = get_file_path_string(
        relative_path_from_project_root=args.path,
        file_name="bf_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_bf,
        weights_or_bias_list=bias_list,
        nbits=current_nbits,
        name="bf",
        index=1,
    )
    file_path_bg = get_file_path_string(
        relative_path_from_project_root=args.path,
        file_name="bg_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_bg,
        weights_or_bias_list=bias_list,
        nbits=current_nbits,
        name="bg",
        index=2,
    )
    file_path_bo = get_file_path_string(
        relative_path_from_project_root=args.path,
        file_name="bo_rom.vhd",
    )
    generate_rom_file(
        file_path=file_path_bo,
        weights_or_bias_list=bias_list,
        nbits=current_nbits,
        name="bo",
        index=3,
    )

    ## generate sigmoid and tanh activation source files ##
    file_path_sigmoid = get_file_path_string(
        relative_path_from_project_root=args.path,
        file_name="sigmoid.vhd",
    )

    with open(file_path_sigmoid, "w") as writer:
        sigmoid = Sigmoid(
            data_width=current_nbits,
            frac_width=current_frac_bits,
            x=np.linspace(-2.5, 2.5, 256),
        )
        sigmoid_code = sigmoid()
        for line in sigmoid_code:
            writer.write(line + "\n")

    file_path_tanh = get_file_path_string(
        relative_path_from_project_root=args.path,
        file_name="tanh.vhd",
    )

    with open(file_path_tanh, "w") as writer:
        tanh = Tanh(
            data_width=current_nbits,
            frac_width=current_frac_bits,
            x=np.linspace(-1, 1, 256),
        )
        tanh_code = tanh()
        for line in tanh_code:
            writer.write(line + "\n")

    ### generate testbench file for use-case ###
    file_path_testbench = get_file_path_string(
        relative_path_from_project_root=args.path,
        file_name="lstm_cell_tb.vhd",
    )

    with open(file_path_testbench, "w") as writer:
        lstm_cell = LSTMCellTestBench(
            data_width=current_nbits,
            frac_width=current_frac_bits,
            input_size=current_input_size,
            hidden_size=current_hidden_size,
            test_x_h_data=x_h_test_input,
            test_c_data=c_test_input,
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

    ### copy static files ###
    for filename in os.listdir(ROOT_DIR + "/vhd_files/static_files/"):
        shutil.copy(
            ROOT_DIR + "/vhd_files/static_files/" + filename,
            ROOT_DIR + "/" + args.path,
        )
