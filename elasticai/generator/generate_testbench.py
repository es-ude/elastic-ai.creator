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


def write_entity(entity_name):
    return """entity {entity_name}_tb is
    port ( clk: out std_logic);
end entity ; -- {entity_name}_tb
\n""".format(entity_name=entity_name)


def write_architecture_header(architecture_name, component_name):
    return """architecture {architecture_name} of {component_name}_tb is
\n""".format(architecture_name=architecture_name, component_name=component_name)


def write_component(data_width, frac_width, variables_dict):
    variables_list_str = ""
    for variable in variables_dict:
        variables_list_str = variables_list_str + "            {variable} : {io} signed(DATA_WIDTH-1 downto 0);\n".format(variable=variable, io=variables_dict[variable])
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


def write_signal_definitions(signal_dict):
    signal_dict_str = ""
    for signal in signal_dict:
        signal_dict_str = signal_dict_str + "    signal " + signal + " : " + signal_dict[signal] + ";\n"
    return """    ------------------------------------------------------------
    -- Testbench Internal Signals
    ------------------------------------------------------------\n""" + signal_dict_str + "\n"


def write_clock_process():
    return """begin

    clock_process : process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process; -- clock_process
\n"""


def write_uut(mapping_dict):
    mapping_dict_str = ""
    for mapping in mapping_dict:
        mapping_dict_str = mapping_dict_str + "    " + mapping + " => " + mapping_dict[mapping] + ",\n"
    # remove last comma and linebreak and add linebreak again
    return """    uut: sigmoid
    port map (\n""" + mapping_dict_str[:-2] + "\n    );\n\n"


def write_test_process_header():
    return """test_process: process is
    begin
        Report "======Simulation start======" severity Note;
\n"""


def write_test_process_end():
    return """        
        -- if there is no error message, that means all test case are passed.
        report "======Simulation Success======" severity Note;
        report "Please check the output message." severity Note;
        
        -- wait forever
        wait;
        
    end process; -- test_process
\n"""


def write_architecture_end(architecture_name):
    return """end {architecture_name} ; -- {architecture_name}
\n""".format(architecture_name=architecture_name)
