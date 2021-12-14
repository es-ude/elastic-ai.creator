import math
import numpy as np
from os import path


def write_libraries(math_lib=False):
    lib_string = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions
"""
    if math_lib:
        lib_string = lib_string + "use ieee.math_real.all;\n"
    else:
        lib_string = lib_string + "\n"
    return lib_string


def write_entity(entity_name, data_width, frac_width):
    return """entity {entity_name} is
    generic (
            DATA_WIDTH: integer := {data_width};
            FRAC_WIDTH: integer := {frac_width};
    );
    port (
     x: in singed(DATA_WIDTH-1 downto 0)
     y: out singed(DATA_WIDTH-1 downto 0)
     );
end {entity_name};
\n""".format(entity_name=entity_name, data_width=data_width, frac_width=frac_width)


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
    return """end {architecture_name} ; -- {architecture_name}
    """.format(architecture_name=architecture_name)


def write_process(component_name):
    return """begin 
    {component_name}_process:process(x)
    variable int_x: integer := 0;
    begin
        int_x := to_integer(x);
        
        {generate_process}
    end process;                
    \n""".format(component_name=component_name, generate_process=_generate_sigmoid_process())


# FIXME : try to make it general maybe with parameters , without nested Function !
def _generate_sigmoid_process():
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    # contains the new lines
    lines = []
    x_list = np.linspace(-5, 5, 66)  # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
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
            lines.append("\ty <= \"" + '{0:016b}'.format(int(256 * sigmoid(x_list[i - 1]))) + "\"; -- " + str(
                int(256 * sigmoid(x_list[i - 1]))))
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
