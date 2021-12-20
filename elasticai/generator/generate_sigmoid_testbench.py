from elasticai.generator.generate_testbench import write_libraries, write_entity, write_architecture_header, \
    write_component, write_signal_definitions, write_clock_process, write_uut, write_test_process_header, \
    write_test_process_end, write_architecture_end, write_begin_architecture
from elasticai.generator.generate_specific_testprocess import write_function_test_process_for_one_input_results_in_one_output


def main(path_to_testbench: str = '../testbench/') -> None:
    """
    generates the vhd testbench file in the testbench folder for the sigmoid function
    Args:
        path_to_testbench: path where testbench is located, default in ../testbench/
    Returns:
        None
    """
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

    with open(path_to_testbench + test_bench_file_name, 'w') as f:
        f.write(write_libraries())
        f.write(write_entity(entity_name=component_name))
        f.write(write_architecture_header(architecture_name=architecture_name, component_name=component_name))
        f.write(write_component(component_name=component_name, data_width=data_width, frac_width=frac_width, variables_dict={
            "x": "in signed(DATA_WIDTH-1 downto 0)",
            "y": "out signed(DATA_WIDTH-1 downto 0)"}))
        f.write(write_signal_definitions(signal_dict={
            "clk_period": "time := 1 ns",
            "test_input": "signed(16-1 downto 0):=(others=>'0')",
            "test_output": "signed(16-1 downto 0)"
        }))
        f.write(write_begin_architecture())
        f.write(write_clock_process())
        f.write(write_uut(component_name=component_name, mapping_dict={"x": "test_input", "y": "test_output"}))
        f.write(write_test_process_header())
        f.write(write_function_test_process_for_one_input_results_in_one_output(inputs=inputs, outputs=outputs, input_name="test_input", output_name="test_output"))
        f.write(write_test_process_end())
        f.write(write_architecture_end(architecture_name=architecture_name))


if __name__ == '__main__':
    main()
