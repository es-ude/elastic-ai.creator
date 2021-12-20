from os import path
from elasticai.generator.generator_functions import *


def write_libraries(math_lib=False, work_lib=False) -> str:
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
    return lib_string + '\n'


def write_entity(entity_name: str, data_width: int, frac_width: int, variables_dict: dict) -> str:
    variables_list_str = ""
    for variable in variables_dict:
        variables_list_str = variables_list_str + """            {variable} : {io} signed(DATA_WIDTH-1 downto 0);\n""".format(
            variable=variable, io=variables_dict[variable])
    return """entity {entity_name} is
    generic (
            DATA_WIDTH: integer := {data_width};
            FRAC_WIDTH: integer := {frac_width}
    );
    port (\n""".format(entity_name=entity_name, data_width=data_width, frac_width=frac_width) \
           + variables_list_str[:-2] \
           + """\n\t\t);
end {entity_name};
\n""".format(entity_name=entity_name)


def write_architecture_header(architecture_name: str, component_name: str) -> str:
    return """architecture {architecture_name} of {component_name} is
\n""".format(architecture_name=architecture_name, component_name=component_name)


def write_component(data_width: any, frac_width: any, component_name: str, variables_dict: dict) -> str:
    variables_list_str = ""
    for variable in variables_dict:
        variables_list_str = variables_list_str + "            {variable} : {io} signed(DATA_WIDTH-1 downto 0);\n".format(
            variable=variable, io=variables_dict[variable])
    # remove last linebreak and semicolon
    return """    component {component_name} is
        generic (
                DATA_WIDTH : integer := {data_width};
                FRAC_WIDTH : integer := {frac_width}
            );
        port (\n""".format(data_width=data_width, frac_width=frac_width,
                           component_name=component_name) + variables_list_str[:-2] + """
        );
    end component {component_name};
\n""".format(component_name=component_name)


def write_architecture_end(architecture_name: str) -> str:
    return """end {architecture_name};
    """.format(architecture_name=architecture_name)


def write_process(component_name: str, process_name) -> str:
    # process_name is a function in a generator_functions
    return """begin 
    {component_name}_process:process(x)
    variable int_x: integer := 0;
    begin
        int_x := to_integer(x);
        
        {generate_process}
    end process;                
    \n""".format(component_name=component_name, generate_process=process_name)


# TODO : I may don't need it at all
def get_path_file(folder_name: str, file_name: str) -> str:
    base_path = path.dirname(__file__)
    path_file = path.abspath(path.join(base_path, "..", folder_name, file_name))
    return path_file


def write_lstm_signals_definition(data_width: int, frac_width: int) -> str:
    return generate_vhdl_codes_for_parameters_of_an_lstm_cell(data_width, frac_width)


def write_input_gate(without_activation: str, with_activation: str) -> str:
    string = """-- Intermediate results
-- Input gate without/with activation
-- i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi})\n""" + (
        """signal {without_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
        signal {with_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');\n\n""").format(
        without_activation=without_activation, with_activation=with_activation)
    return string


def write_forget_gate(without_activation: str, with_activation: str) -> str:
    string = """-- Forget gate without/with activation
-- f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf})\n""" + (
        """signal {without_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
signal {with_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');\n\n""").format(
        without_activation=without_activation, with_activation=with_activation)
    return string


def write_cell_gate(without_activation: str, with_activation: str) -> str:
    string = """-- Cell gate without/with activation
-- g = \\tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg})\n""" + (
        """signal {without_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
signal {with_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');\n\n""".format(
            without_activation=without_activation, with_activation=with_activation))
    return string


def write_output_gate(without_activation: str, with_activation: str) -> str:
    string = """-- Output gate without/with activation
-- o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho})\n""" + (
        """signal {without_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
        signal {with_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');\n\n""".format(
            without_activation=without_activation, with_activation=with_activation))
    return string


def write_new_cell_state(without_activation: str, with_activation: str) -> str:
    return """-- new_cell_state without/with activation
-- c' = f * c + i * g 
signal {without_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
signal {with_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');

-- h' = o * \\tanh(c')
signal h_new : signed(DATA_WIDTH-1 downto 0):=(others=>'0');\n\n\n""".format(without_activation=without_activation,
                                                                             with_activation=with_activation)


def write_architecture_begin() -> str: return "begin\n"


def write_architecture_signals_definition(variable_list: list) -> str:
    string = ""
    for variable in variable_list:
        string += """    signal {variable} :signed(DATA_WIDTH-1 downto 0);\n""".format(variable=variable)
    return string


# TODO: it should be more general !
def write_mac_async_architecture_behavior() -> str:
    return """    -- behavior: y=w1*x1+w2*x2+b
    product_1 <= shift_right((x1 * w1), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);
    product_2 <= shift_right((x2 * w2), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);
    y <= product_1 + product_2 + b;\n"""


def write_signal_map(variables_dict: dict) -> str:
    string = "    -- signals will be output to the cell\n"
    for variable in variables_dict:
        string += """    {val1} <= {val2};\n""".format(val1=variable, val2=variables_dict[variable])
    return string + "\n"


def write_port_map(map_name: str, component_name: str, signals) -> str:
    # the typ of signals can be list or dict !!
    string = """    {map_name}: {component_name}\n    port map (\n""".format(map_name=map_name,
                                                                             component_name=component_name)
    if type(signals) == list:
        for signal in signals:
            string += """        {signal},\n""".format(signal=signal)
    else:
        for signal in signals:
            string += """        {val1} => {val2},\n""".format(val1=signal, val2=signals[signal])
    # remove the last comma and new line
    string = string.removesuffix(",\n") + "\n"
    # add close bracket
    string += "    );\n\n"
    return string


# TODO: change this
def write_lstm_process() -> str:
    return """    H_OUT_PROCESS: process(o,c_new)
    begin
        h_new <= shift_right((o*c_new), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);
    end process;\n\n"""
