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


def write_test_process(inputs, outputs):
    test = ""
    if len(inputs) == len(outputs):
        for i in range(len(inputs)):
            test = test + """        test_input <=  to_signed({input},16);
        wait for 1*clk_period;
        report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
        assert test_output={output} report "The test case {input} fail" severity failure;
\n""".format(input=inputs[i], output=outputs[i])
        return test
    else:
        raise TypeError(f"inputs length {len(inputs)} is different to outputs length {len(outputs)}.")


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


def main():
    component_name = "sigmoid"
    test_bench_file_name = component_name + "_tb.vhd"
    architecture_name = "behav"

    data_width = 16
    frac_width = 8

    # x, y = sigmoid(data_width, frac_width)
    # Note, the two array below, is generated based on data_width and frac_width
    # excitation signals, as test inputs signal
    inputs = [-1281, -1000, -500]
    # expected signal, as test reference output signal
    outputs = [0, 4, 28]

    # FIXME: change back
    with open('../testbench/generated_' + test_bench_file_name, 'w') as f:
        f.write(write_libraries())
        f.write(write_entity(entity_name=component_name))
        f.write(write_architecture_header(architecture_name=architecture_name, component_name=component_name))
        f.write(write_component(data_width=data_width, frac_width=frac_width, variables_dict={"x": "in", "y": "out"}))
        f.write(write_signal_definitions(signal_dict={
            "clk_period": "time := 1 ns",
            "test_input": "signed(16-1 downto 0):=(others=>'0')",
            "test_output": "signed(16-1 downto 0)"
        }))
        f.write(write_clock_process())
        f.write(write_uut(mapping_dict={"x": "test_input", "y": "test_output"}))
        f.write(write_test_process_header())
        f.write(write_test_process(inputs, outputs))
        f.write(write_test_process_end())
        f.write(write_architecture_end(architecture_name=architecture_name))


if __name__ == '__main__':
    main()
