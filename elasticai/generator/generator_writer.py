import math
from os import path
import torch.nn as nn
from elasticai.creator.layers import QLSTMCell

def write_libraries(math_lib=False, work_lib=False):
    lib_string = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions
"""
    if math_lib:
        lib_string = lib_string + "use ieee.math_real.all;\n"
    if work_lib:
        lib_string = lib_string + "LIBRARY work;\nuse work.all;\n"
    else:
        lib_string = lib_string + "\n"
    return lib_string


def write_entity(entity_name, data_width, frac_width, is_y=False):
    if is_y:
        ports = "     y : out signed(DATA_WIDTH-1 downto 0)"
    else:
        ports = """c_in : in signed(DATA_WIDTH-1 downto 0);
    h_in : in signed(DATA_WIDTH-1 downto 0);

    c_out: out signed(DATA_WIDTH-1 downto 0);
    h_out: out signed(DATA_WIDTH-1 downto 0)
        """
    return """entity {entity_name} is
    generic (
            DATA_WIDTH: integer := {data_width};
            FRAC_WIDTH: integer := {frac_width}
    );
    port (
     x : in signed(DATA_WIDTH-1 downto 0);
    {ports}
     );
end {entity_name};
\n""".format(entity_name=entity_name, data_width=data_width, frac_width=frac_width, ports=ports)


def write_architecture_header(architecture_name, component_name):
    return """architecture {architecture_name} of {component_name} is
\n""".format(architecture_name=architecture_name, component_name=component_name)


#  FIXME : maybe I need to change the def !!
def write_component(data_width, frac_width, variables_dict):
    variables_list_str = ""
    for variable in variables_dict:
        variables_list_str = variables_list_str + "            {variable} : {io} signed(DATA_WIDTH-1 downto 0);\n".format(
            variable=variable, io=variables_dict[variable])
    # remove last linebreak and semicolon
    return """    component sigmoid is
        generic (
                DATA_WIDTH : integer := {data_width};
                FRAC_WIDTH : integer := {frac_width}
            );
        port (\n""".format(data_width=data_width, frac_width=frac_width) + variables_list_str[:-2] + """
        );
    end component;
\n"""


def write_architecture_end(architecture_name):
    return """end {architecture_name};
    """.format(architecture_name=architecture_name)


def write_process(component_name, process_name):
    return """begin 
    {component_name}_process:process(x)
    variable int_x: integer := 0;
    begin
        int_x := to_integer(x);
        
        {generate_process}
    end process;                
    \n""".format(component_name=component_name, generate_process=process_name)


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


def get_path_file(folder_name: str, file_name: str) -> str:
    base_path = path.dirname(__file__)
    path_file = path.abspath(path.join(base_path, "..", folder_name, file_name))
    return path_file


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


def _elasticaicreator_lstm():
    return QLSTMCell(1, 1, state_quantizer=nn.Identity(), weight_quantizer=nn.Identity())


def generate_vhdl_codes_for_parameters_of_an_lstm_cell(lstm_singal_cell):

    ensure_reproducibility()



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

    for name, param in lstm_singal_cell.named_parameters():
        if name == "weight_ih":
            print(to_vhdl_parameter(param[0], frac_width, nbits, name_parameter="W_ii",
                                    vhdl_prefix="signal wii: signed(DATA_WIDTH-1 downto 0) := "))
            print(to_vhdl_parameter(param[1], frac_width, nbits, name_parameter="W_if",
                                    vhdl_prefix="signal wif : signed(DATA_WIDTH-1 downto 0) := "))
            print(to_vhdl_parameter(param[2], frac_width, nbits, name_parameter="W_ig",
                                    vhdl_prefix="signal wig : signed(DATA_WIDTH-1 downto 0) := "))
            print(to_vhdl_parameter(param[3], frac_width, nbits, name_parameter="W_io",
                                    vhdl_prefix="signal wio : signed(DATA_WIDTH-1 downto 0) := "))
        elif name == "weight_hh":
            print(to_vhdl_parameter(param[0], frac_width, nbits, name_parameter="W_hi",
                                    vhdl_prefix="signal whi : signed(DATA_WIDTH-1 downto 0) := "))
            print(to_vhdl_parameter(param[1], frac_width, nbits, name_parameter="W_hf",
                                    vhdl_prefix="signal whf : signed(DATA_WIDTH-1 downto 0) := "))
            print(to_vhdl_parameter(param[2], frac_width, nbits, name_parameter="W_hg",
                                    vhdl_prefix="signal whg : signed(DATA_WIDTH-1 downto 0) := "))
            print(to_vhdl_parameter(param[3], frac_width, nbits, name_parameter="W_ho",
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
            print("should not come to here.")

    print(to_vhdl_parameter(b_ii + b_hi, frac_width, nbits, name_parameter="b_ii + b_hi",
                            vhdl_prefix="signal bi : signed(DATA_WIDTH-1 downto 0) := "))
    print(to_vhdl_parameter(b_if + b_hf, frac_width, nbits, name_parameter="b_if + b_hf",
                            vhdl_prefix="signal bf : signed(DATA_WIDTH-1 downto 0) := "))
    print(to_vhdl_parameter(b_ig + b_hg, frac_width, nbits, name_parameter="b_ig + b_hg",
                            vhdl_prefix="signal bg : signed(DATA_WIDTH-1 downto 0) := "))
    print(to_vhdl_parameter(b_io + b_ho, frac_width, nbits, name_parameter="b_io + b_ho",
                            vhdl_prefix="signal bo : signed(DATA_WIDTH-1 downto 0) := "))

