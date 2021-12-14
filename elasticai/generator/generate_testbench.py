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


def write_component(data_width, frac_width):
    return """    component sigmoid is
        generic (
                DATA_WIDTH : integer := {};
                FRAC_WIDTH : integer := {}
            );
        port (
            x : in signed(DATA_WIDTH-1 downto 0);
            y: out signed(DATA_WIDTH-1 downto 0)
        );
    end component;
\n""".format(data_width, frac_width)


def write_signal_definitions():
    return """    ------------------------------------------------------------
    -- Testbench Internal Signals
    ------------------------------------------------------------
    signal clk_period : time := 1 ns;
    signal test_input : signed(16-1 downto 0):=(others=>'0');
    signal test_output : signed(16-1 downto 0);
\n"""


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


def write_utt():
    return """    utt: sigmoid
    port map (
    x => test_input,
    y => test_output
    );
\n"""


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
        f.write(write_component(data_width=data_width, frac_width=frac_width))
        f.write(write_signal_definitions())
        f.write(write_clock_process())
        f.write(write_utt())
        f.write(write_test_process_header())
        f.write(write_test_process(inputs, outputs))
        f.write(write_test_process_end())
        f.write(write_architecture_end(architecture_name=architecture_name))


if __name__ == '__main__':
    main()
