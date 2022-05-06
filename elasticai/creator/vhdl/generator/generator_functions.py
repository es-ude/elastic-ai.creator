import math
import random
from functools import partial
from os import path
from typing import Dict, List, Iterable

import numpy as np
import torch
import torch.nn as nn

from elasticai.creator.layers import QLSTMCell
from elasticai.creator.vhdl.language import CodeGenerator
from elasticai.creator.vhdl.number_representations import (
    FloatToSignedFixedPointConverter,
    FloatToBinaryFixedPointStringConverter,
    BitVector,
    FloatToHexFixedPointStringConverter,
)
from paths import ROOT_DIR


def vhdl_add_assignment(code: list, line_id: str, value: str, comment=None) -> None:
    new_code_fragment = f'{line_id} <= "{value}";'
    if comment is not None:
        new_code_fragment += f" -- {comment}"
    code.append(new_code_fragment)


def precomputed_scalar_function_process(x_list, y_list) -> CodeGenerator:
    """
        returns the string of a lookup table
    Args:
        y_list : output List contains integers
        x_list: input List contains integers
    Returns:
        String of lookup table (if/elsif statements for vhdl file)
    """
    as_signed_fixed_point = FloatToSignedFixedPointConverter(
        bits_used_for_fraction=8, strict=False
    )
    as_binary_string = FloatToBinaryFixedPointStringConverter(
        total_bit_width=16, as_signed_fixed_point=as_signed_fixed_point
    )
    x_list.sort()
    lines = []
    if len(x_list) == 0 and len(y_list) == 1:
        vhdl_add_assignment(
            code=lines,
            line_id="y",
            value=as_binary_string(y_list[0]),
        )
    elif len(x_list) != len(y_list) - 1:
        raise ValueError(
            "x_list has to be one element shorter than y_list, but x_list has {} elements and y_list {} elements".format(
                len(x_list), len(y_list)
            )
        )
    else:
        smallest_possible_output = y_list[0]
        biggest_possible_output = y_list[-1]

        # first element
        for x in x_list[:1]:
            lines.append("if int_x<{0} then".format(as_signed_fixed_point(x)))
            vhdl_add_assignment(
                code=lines,
                line_id="y",
                value=as_binary_string(smallest_possible_output),
                comment=as_signed_fixed_point(smallest_possible_output),
            )
            lines[-1] = "\t" + lines[-1]
        for current_x, current_y in zip(x_list[1:], y_list[1:-1]):
            lines.append(
                "elsif int_x<{0} then".format(as_signed_fixed_point(current_x))
            )
            vhdl_add_assignment(
                code=lines,
                line_id="y",
                value=as_binary_string(current_y),
                comment=as_signed_fixed_point(current_y),
            )
            lines[-1] = "\t" + lines[-1]
        # last element only in y
        for _ in y_list[-1:]:
            lines.append("else")
            vhdl_add_assignment(
                code=lines,
                line_id="y",
                value=as_binary_string(biggest_possible_output),
                comment=as_signed_fixed_point(biggest_possible_output),
            )
            lines[-1] = "\t" + lines[-1]
        if len(lines) != 0:
            lines.append("end if;")
    # build the string block
    yield lines[0]
    for line in lines[1:]:
        yield line


def precomputed_logic_function_process(
    x_list: List[List[BitVector]],
    y_list: List[List[BitVector]],
) -> CodeGenerator:
    """
        returns the string of a lookup table where the value of the input exactly equals x
    Args:
        y_list : output List contains integers
        x_list: input List contains integers
    Returns:
        String of lookup table (if/elsif statements for vhdl file)
    """

    lines = []
    if len(x_list) != len(y_list):
        raise ValueError(
            "x_list has to be the same length as y_list, but x_list has {} elements and y_list {} elements".format(
                len(x_list), len(y_list)
            )
        )
    else:
        x_bit_vectors = []
        y_bit_vectors = []
        for x_element, y_element in zip(x_list, y_list):
            x_bit_vectors.append("".join(list(map(lambda x: x.__repr__(), x_element))))
            y_bit_vectors.append("".join(list(map(lambda x: x.__repr__(), y_element))))
        # first element
        iterator = zip(x_bit_vectors, y_bit_vectors)
        first = next(iterator)
        lines.append(f'y <="{first[1]}" when x="{first[0]}" else')
        for x, y in iterator:
            lines.append(f'"{y}" when x="{x}" else\n')
        lines[-1] = lines[-1][:-5] + ";"
    # build the string block
    yield lines[0]
    for line in lines[1:]:
        yield line


def float_array_to_int(float_array: np.array, frac_bits: int) -> np.array:
    """
    converts an array with floating point numbers into an array with integers
    Args:
        float_array (NDArray[Float32]): array with floating point numbers
        frac_bits (int): number of fraction bits
    Returns:
        array with integer numbers
    """
    int_list = []
    floats_to_signed_fixed_point_converter = FloatToSignedFixedPointConverter(
        bits_used_for_fraction=frac_bits, strict=False
    )
    for element in float_array:
        int_list.append(floats_to_signed_fixed_point_converter(element))
    return np.array(int_list)


def float_array_to_hex_string(
    float_array: np.array, frac_bits: int, number_of_bits: int
) -> list[str]:
    list_with_hex_representation = []
    for element in float_array:
        floats_to_signed_fixed_point_converter = FloatToSignedFixedPointConverter(
            bits_used_for_fraction=frac_bits, strict=False
        )
        # convert to hex
        float_to_hex_fixed_point_string_converter = FloatToHexFixedPointStringConverter(
            total_bit_width=number_of_bits,
            as_signed_fixed_point=floats_to_signed_fixed_point_converter,
        )
        list_with_hex_representation.append(
            float_to_hex_fixed_point_string_converter(element)
        )
    # fill up to address width with zeros
    addr_width = math.ceil(math.log2(len(list_with_hex_representation)))
    if addr_width == 0:
        addr_width = 1
    number_of_hex_numbers = int(number_of_bits / 4)
    zero_string = "".join("0" for _ in range(number_of_hex_numbers))
    for index in range(2**addr_width - len(list_with_hex_representation)):
        list_with_hex_representation.append(zero_string)
    return list_with_hex_representation


def _to_vhdl_parameter(
    f_val: float, frac_width: int, nbits: int, signal_name: str
) -> Dict:
    """
        returns a Dictionary of one signal and his definition
    Args:
        f_val (float): a floating point number
        nbits(int): the number of bits
        frac_width(int): the number of bits for the fraction
    Returns:
        String of a hexadecimal value for an integer
    """
    floats_to_signed_fixed_point_converter = FloatToSignedFixedPointConverter(
        bits_used_for_fraction=frac_width, strict=False
    )
    # convert to hex
    float_to_hex_fixed_point_string_converter = FloatToHexFixedPointStringConverter(
        total_bit_width=nbits,
        as_signed_fixed_point=floats_to_signed_fixed_point_converter,
    )
    return {
        str(signal_name): 'X"' + float_to_hex_fixed_point_string_converter(f_val) + '"'
    }


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
    Args:
        data_width (int): the width of the data
        frac_width (int): the fraction part of data_width
    Returns:
        Dict of the signal names and their definitions
    """
    dict_of_signals = {}
    # define the lstm cell
    _ensure_reproducibility()
    lstm_single_cell = _elastic_ai_creator_lstm()

    # weight_ih_l[k] : `(W_ii|W_if|W_ig|W_io)`
    # weight_hh_l[k] : `(W_hi|W_hf|W_hg|W_ho)`
    # bias_ih_l[k] :  `(b_ii|b_if|b_ig|b_io)`
    # bias_hh_l[k] :  `(W_hi|W_hf|W_hg|b_ho)`
    b_ii = 0
    b_if = 0
    b_ig = 0
    b_io = 0

    b_hi = 0
    b_hf = 0
    b_hg = 0
    b_ho = 0

    floats_to_signed_fixed_point_converter = FloatToSignedFixedPointConverter(
        bits_used_for_fraction=frac_width, strict=False
    )
    float_to_hex_fixed_point_string_converter = FloatToHexFixedPointStringConverter(
        total_bit_width=data_width,
        as_signed_fixed_point=floats_to_signed_fixed_point_converter,
    )

    def update_signals(signal_names: Iterable[str], signal_values: Iterable[float]):
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
        elif name == "bias_ih":
            b_ii = param[0]
            b_if = param[1]
            b_ig = param[2]
            b_io = param[3]
        elif name == "bias_hh":
            b_hi = param[0]
            b_hf = param[1]
            b_hg = param[2]
            b_ho = param[3]
        else:
            dict_of_signals.update("should not come to here.")

    dict_of_signals.update(
        _to_vhdl_parameter(
            b_ii + b_hi,
            frac_width,
            data_width,
            signal_name="bi",
        )
    )
    dict_of_signals.update(
        _to_vhdl_parameter(
            b_if + b_hf,
            frac_width,
            data_width,
            signal_name="bf",
        )
    )
    dict_of_signals.update(
        _to_vhdl_parameter(
            b_ig + b_hg,
            frac_width,
            data_width,
            signal_name="bg",
        )
    )
    dict_of_signals.update(
        _to_vhdl_parameter(
            b_io + b_ho,
            frac_width,
            data_width,
            signal_name="bo",
        )
    )

    return dict_of_signals


def get_file_path_string(
    file_name: str,
    relative_path_from_project_root: str,
) -> str:
    """
    returns String of a file path
    Args:
        file_name (str): the name of the file
        relative_path_from_project_root (str): path where file should be located
    Returns:
        string of the full path to a given filename
    """
    return path.join(ROOT_DIR, relative_path_from_project_root, file_name)
