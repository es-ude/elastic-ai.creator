import math
import random
import torch
import torch.nn as nn
from elasticai.creator.layers import QLSTMCell


def tanh_process(x_list):
    def _bindigits(n, bits):
        s = bin(n & int("1" * bits, 2))[2:]
        return ("{0:0>%s}" % bits).format(s)

    lines = []

    frac_bits = 8
    one = 2 ** frac_bits

    for i in range(len(x_list)):
        if i == 0:
            lines.append("if int_x<" + str(int(x_list[0] * one)) + " then")
            lines.append("\ty <= \"" + _bindigits(-1 * one, 16) + "\"; -- " + str(-1 * one))
        elif i == (len(x_list) - 1):
            lines.append("else")
            lines.append("\ty <= \"" + '{0:016b}'.format(one) + "\"; -- " + str(one))
        else:
            lines.append("elsif int_x<" + str(int(x_list[i] * one)) + " then")

            lines.append("\ty <= \"" + _bindigits(int(256 * math.tanh(x_list[i - 1])), 16) + "\"; -- " + str(
                int(256 * math.tanh(x_list[i - 1]))))

    lines.append("end if;")
    # build the string block and add new line + 2 tabs
    string = ""
    for line in lines:
        string = string + line + "\n" + "\t" + "\t"
    return string


def sigmoid_process(x_list):
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
            lines.append("\ty <= \"" + '{0:016b}'.format(zero) + "\"; -- " + str(zero))
        elif i == (len(x_list) - 1):
            lines.append("else")
            lines.append("\ty <= \"" + '{0:016b}'.format(one) + "\"; -- " + str(one))
        else:
            lines.append("elsif int_x<" + str(int(x_list[i] * one)) + " then")
            lines.append("\ty <= \"" + '{0:016b}'.format(int(256 * _sigmoid(x_list[i - 1]))) + "\"; -- " + str(
                int(256 * _sigmoid(x_list[i - 1]))))
    lines.append("end if;")
    # build the string block and add new line + 2 tabs
    string = ""
    for line in lines:
        string = string + line + "\n" + "\t" + "\t"
    return string


def _int_to_hex(val, nbits):
    if val < 0:
        return hex((val + (1 << nbits)) % (1 << nbits))
    else:
        return "{0:#0{1}x}".format(val, 2 + int(nbits / 4))


def _floating_to_fixed_point_int(f_val, frac_width):
    return str(int(f_val * (2 ** frac_width)))


def _floating_to_hex(f_val, frac_width, nbits):
    int_val = int(f_val * (2 ** frac_width))
    return _int_to_hex(int_val, nbits)


def _to_vhdl_parameter(f_val, frac_width, nbits, name_parameter=None, vhdl_prefix=None):
    hex_str = _floating_to_hex(f_val, frac_width, nbits)
    hex_str_without_prefix = hex_str[2:]
    if name_parameter == None:
        return hex_str_without_prefix
    else:
        return "" + vhdl_prefix + "X\"" + hex_str_without_prefix + "\"; -- " + name_parameter


def _pytorch_lstm():
    return nn.LSTMCell(1, 1)


def elasticaicreator_lstm():
    return QLSTMCell(1, 1, state_quantizer=nn.Identity(), weight_quantizer=nn.Identity())


def _ensure_reproducibility():
    torch.manual_seed(0)
    random.seed(0)


def generate_vhdl_codes_for_parameters_of_an_lstm_cell(data_width, frac_width):
    lines = []
    # define the lstm cell
    _ensure_reproducibility()
    lstm_single_cell = elasticaicreator_lstm()

    # print the parameters in the vhdl format
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
            lines.append(_to_vhdl_parameter(param[0], frac_width, data_width, name_parameter="W_ii",
                                            vhdl_prefix="signal wii: signed(DATA_WIDTH-1 downto 0) := "))
            lines.append(_to_vhdl_parameter(param[1], frac_width, data_width, name_parameter="W_if",
                                            vhdl_prefix="signal wif : signed(DATA_WIDTH-1 downto 0) := "))
            lines.append(_to_vhdl_parameter(param[2], frac_width, data_width, name_parameter="W_ig",
                                            vhdl_prefix="signal wig : signed(DATA_WIDTH-1 downto 0) := "))
            lines.append(_to_vhdl_parameter(param[3], frac_width, data_width, name_parameter="W_io",
                                            vhdl_prefix="signal wio : signed(DATA_WIDTH-1 downto 0) := "))
        elif name == "weight_hh":
            lines.append(_to_vhdl_parameter(param[0], frac_width, data_width, name_parameter="W_hi",
                                            vhdl_prefix="signal whi : signed(DATA_WIDTH-1 downto 0) := "))
            lines.append(_to_vhdl_parameter(param[1], frac_width, data_width, name_parameter="W_hf",
                                            vhdl_prefix="signal whf : signed(DATA_WIDTH-1 downto 0) := "))
            lines.append(_to_vhdl_parameter(param[2], frac_width, data_width, name_parameter="W_hg",
                                            vhdl_prefix="signal whg : signed(DATA_WIDTH-1 downto 0) := "))
            lines.append(_to_vhdl_parameter(param[3], frac_width, data_width, name_parameter="W_ho",
                                            vhdl_prefix="signal who : signed(DATA_WIDTH-1 downto 0) := "))
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
            lines.append("should not come to here.")

    lines.append(_to_vhdl_parameter(b_ii + b_hi, frac_width, data_width, name_parameter="b_ii + b_hi",
                                    vhdl_prefix="signal bi : signed(DATA_WIDTH-1 downto 0) := "))
    lines.append(_to_vhdl_parameter(b_if + b_hf, frac_width, data_width, name_parameter="b_if + b_hf",
                                    vhdl_prefix="signal bf : signed(DATA_WIDTH-1 downto 0) := "))
    lines.append(_to_vhdl_parameter(b_ig + b_hg, frac_width, data_width, name_parameter="b_ig + b_hg",
                                    vhdl_prefix="signal bg : signed(DATA_WIDTH-1 downto 0) := "))
    lines.append(_to_vhdl_parameter(b_io + b_ho, frac_width, data_width, name_parameter="b_io + b_ho",
                                    vhdl_prefix="signal bo : signed(DATA_WIDTH-1 downto 0) := "))
    # build the string of lines
    string = ""
    for line in lines:
        string = string + line + '\n' + '\t'
    return """   -- -- signals -- -- 
    {string} 
    -- -- signals -- --\n\n""".format(string=string)
