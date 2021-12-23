import math
import random
import torch
import torch.nn as nn
from elasticai.creator.layers import QLSTMCell
from typing import Dict, List


def tanh_process(x_list: List) -> str:
    """
        returns the string of a lookup table
    Args:
        x_list (List): List contains integers
    Returns:
        String of lookup table (if/elsif statements for vhdl file)
    """

    def _bin_digits(n, bits):
        s = bin(n & int("1" * bits, 2))[2:]
        return ("{0:0>%s}" % bits).format(s)

    lines = []

    frac_bits = 8
    one = 2 ** frac_bits

    for i in range(len(x_list)):
        if i == 0:
            lines.append("if int_x<" + str(int(x_list[0] * one)) + " then")
            lines.append(
                '\ty <= "' + _bin_digits(-1 * one, 16) + '"; -- ' + str(-1 * one)
            )
        elif i == (len(x_list) - 1):
            lines.append("else")
            lines.append('\ty <= "' + "{0:016b}".format(one) + '"; -- ' + str(one))
        else:
            lines.append("elsif int_x<" + str(int(x_list[i] * one)) + " then")

            lines.append(
                '\ty <= "'
                + _bin_digits(int(256 * math.tanh(x_list[i - 1])), 16)
                + '"; -- '
                + str(int(256 * math.tanh(x_list[i - 1])))
            )

    lines.append("end if;")
    # build the string block and add new line + 2 tabs
    string = ""
    for line in lines:
        string = string + line + "\n" + "\t" + "\t"
    return string


def sigmoid_process(x_list: List) -> str:
    """
        returns the string of a lookup table
    Args:
        x_list (List): List contains integers
    Returns:
        String of lookup table (if/elsif statements for vhdl file)
    """

    def _sigmoid(x):
        return 1 / (1 + math.exp(-x))

    # contains the new lines
    lines = []

    frac_bits = 8
    zero = 0
    one = 2 ** frac_bits

    for i in range(len(x_list)):
        if i == 0:
            lines.append("if int_x<" + str(int(x_list[0] * one)) + " then")
            lines.append('\ty <= "' + "{0:016b}".format(zero) + '"; -- ' + str(zero))
        elif i == (len(x_list) - 1):
            lines.append("else")
            lines.append('\ty <= "' + "{0:016b}".format(one) + '"; -- ' + str(one))
        else:
            lines.append("elsif int_x<" + str(int(x_list[i] * one)) + " then")
            lines.append(
                '\ty <= "'
                + "{0:016b}".format(int(256 * _sigmoid(x_list[i - 1])))
                + '"; -- '
                + str(int(256 * _sigmoid(x_list[i - 1])))
            )
    lines.append("end if;")
    # build the string block and add new line + 2 tabs
    string = ""
    for line in lines:
        string = string + line + "\n" + "\t" + "\t"
    return string


def _int_to_hex(val: int, nbits: int) -> str:
    """
        returns a string of a hexadecimal value for an integer
    Args:
        val (int): the integer value which will converted in hexadecimal
        nbits(int): the number of bits
    Returns:
        String of a hexadecimal value for an integer
    """
    if val < 0:
        return hex((val + (1 << nbits)) % (1 << nbits))
    else:
        return "{0:#0{1}x}".format(val, 2 + int(nbits / 4))


def _floating_to_hex(f_val: float, frac_width: int, nbits: int) -> str:
    """
        returns a string of a hexadecimal value for a floating point number
    Args:
        f_val (int): the floating point number which will converted to hexadecimal
        nbits(int): the number of bits
        frac_width(int): the number of bits for the fraction
    Returns:
        String of a hexadecimal value for a floating number
    """
    int_val = int(f_val * (2 ** frac_width))
    return _int_to_hex(int_val, nbits)


def _to_vhdl_parameter(
    f_val: float, frac_width: int, nbits: int, name_parameter: str, signal_name: str
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
    hex_str = _floating_to_hex(f_val, frac_width, nbits)
    hex_str_without_prefix = hex_str[2:]

    return {
        str(signal_name): "signed(DATA_WIDTH-1 downto 0) := "
        + 'X"'
        + hex_str_without_prefix
        + '"; -- '
        + name_parameter
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

    for name, param in lstm_single_cell.named_parameters():
        if name == "weight_ih":
            dict_of_signals.update(
                _to_vhdl_parameter(
                    param[0],
                    frac_width,
                    data_width,
                    name_parameter="W_ii",
                    signal_name="wii",
                )
            )
            dict_of_signals.update(
                _to_vhdl_parameter(
                    param[1],
                    frac_width,
                    data_width,
                    name_parameter="W_if",
                    signal_name="wif",
                )
            )
            dict_of_signals.update(
                _to_vhdl_parameter(
                    param[2],
                    frac_width,
                    data_width,
                    name_parameter="W_ig",
                    signal_name="wig",
                )
            )
            dict_of_signals.update(
                _to_vhdl_parameter(
                    param[3],
                    frac_width,
                    data_width,
                    name_parameter="W_io",
                    signal_name="wio",
                )
            )
        elif name == "weight_hh":
            dict_of_signals.update(
                _to_vhdl_parameter(
                    param[0],
                    frac_width,
                    data_width,
                    name_parameter="W_hi",
                    signal_name="whi",
                )
            )
            dict_of_signals.update(
                _to_vhdl_parameter(
                    param[1],
                    frac_width,
                    data_width,
                    name_parameter="W_hf",
                    signal_name="whf",
                )
            )
            dict_of_signals.update(
                _to_vhdl_parameter(
                    param[2],
                    frac_width,
                    data_width,
                    name_parameter="W_hg",
                    signal_name="whg",
                )
            )
            dict_of_signals.update(
                _to_vhdl_parameter(
                    param[3],
                    frac_width,
                    data_width,
                    name_parameter="W_ho",
                    signal_name="who",
                )
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
            name_parameter="b_ii + b_hi",
            signal_name="bi",
        )
    )
    dict_of_signals.update(
        _to_vhdl_parameter(
            b_if + b_hf,
            frac_width,
            data_width,
            name_parameter="b_if + b_hf",
            signal_name="bf",
        )
    )
    dict_of_signals.update(
        _to_vhdl_parameter(
            b_ig + b_hg,
            frac_width,
            data_width,
            name_parameter="b_ig + b_hg",
            signal_name="bg",
        )
    )
    dict_of_signals.update(
        _to_vhdl_parameter(
            b_io + b_ho,
            frac_width,
            data_width,
            name_parameter="b_io + b_ho",
            signal_name="bo",
        )
    )

    return dict_of_signals
