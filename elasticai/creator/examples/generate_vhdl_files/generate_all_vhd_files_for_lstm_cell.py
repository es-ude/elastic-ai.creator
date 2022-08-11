import os
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch

from elasticai.creator.qat.layers import QLSTMCell
from elasticai.creator.resource_utils import copy_file
from elasticai.creator.vhdl.generator_functions_for_lstm import (
    extract_fixed_point_weights_and_bias,
    generate_rom_file,
    inference_model_on_random_data,
)
from elasticai.creator.vhdl.lstm_testbench_generator import LSTMCellTestBench
from elasticai.creator.vhdl.number_representations import float_values_to_fixed_point
from elasticai.creator.vhdl.precomputed_scalar_function import Sigmoid, Tanh
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl

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


def main() -> None:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--path",
        help="path to folder for generated vhd files",
        required=True,
    )
    args = arg_parser.parse_args()
    if not os.path.isdir(args.path):
        os.mkdir(args.path)

    def destination_path(file_name: str) -> str:
        return os.path.join(args.path, file_name)

    # set the current values
    torch.manual_seed(0)
    frac_bits = 8
    total_bits = 16
    input_size = 1
    hidden_size = 20

    to_fp = partial(
        float_values_to_fixed_point, total_bits=total_bits, frac_bits=frac_bits
    )

    lstm_cell = define_lstm_cell(input_size, hidden_size)

    weights_list, bias_list = extract_fixed_point_weights_and_bias(
        lstm_cell, total_bits, frac_bits
    )
    print("weights_list:", [list(map(lambda x: x.to_hex(), w)) for w in weights_list])
    print("bias_list:", [list(map(lambda x: x.to_hex(), b)) for b in bias_list])

    x_h_test_input, c_test_input, h_output = inference_model_on_random_data(
        lstm_cell, total_bits, frac_bits
    )
    print("x_h_test_input:", list(map(lambda x: x.to_hex(), x_h_test_input)))
    print("c_test_input:", list(map(lambda x: x.to_hex(), c_test_input)))
    print("h_output:", list(map(lambda x: x.to_hex(), h_output)))

    # generate source files for use-case

    # generate weight and bias source files
    weight_names = ["wi", "wf", "wg", "wo", "bi", "bf", "bg", "bo"]
    for name, values in zip(weight_names, weights_list + bias_list):
        generate_rom_file(
            file_path=destination_path(f"{name}_rom.vhd"),
            rom_name=f"{name}_rom",
            values=values,
            resource_option="auto",
        )

    # generate sigmoid and tanh activation source files
    file_path_sigmoid = destination_path("sigmoid.vhd")
    with open(file_path_sigmoid, "w") as writer:
        sigmoid = Sigmoid(
            x=to_fp(np.linspace(-2.5, 2.5, 256).tolist()),
        )
        sigmoid_code = sigmoid()
        for line in sigmoid_code:
            writer.write(line + "\n")

    file_path_tanh = destination_path("tanh.vhd")
    with open(file_path_tanh, "w") as writer:
        tanh = Tanh(
            x=to_fp(np.linspace(-1, 1, 256).tolist()),
        )
        tanh_code = tanh()
        for line in tanh_code:
            writer.write(line + "\n")

    # generate testbench file for use-case
    file_path_testbench = destination_path("lstm_cell_tb.vhd")
    with open(file_path_testbench, "w") as writer:
        lstm_cell_tb = LSTMCellTestBench(
            input_size=input_size,
            hidden_size=hidden_size,
            test_x_h_data=x_h_test_input,
            test_c_data=c_test_input,
            h_out=h_output,
            component_name="lstm_cell",
        )
        lstm_cell_code = lstm_cell_tb()
        for line in lstm_cell_code:
            writer.write(line + "\n")

    for name in weight_names:
        format_vhdl(file_path=destination_path(f"{name}_rom.vhd"))

    format_vhdl(file_path=file_path_sigmoid)
    format_vhdl(file_path=file_path_tanh)
    format_vhdl(file_path=file_path_testbench)

    # copy static files
    for file in ["dual_port_2_clock_ram.vhd", "lstm.vhd", "lstm_common.vhd"]:
        copy_file("elasticai.creator.vhdl.templates", file, destination_path(file))


if __name__ == "__main__":
    main()
