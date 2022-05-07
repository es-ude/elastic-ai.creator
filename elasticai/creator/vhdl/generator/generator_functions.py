import random
from os import path
from typing import Dict, List, Iterable

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
