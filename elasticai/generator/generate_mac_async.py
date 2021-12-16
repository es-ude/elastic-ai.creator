from elasticai.generator.generate_testbench import write_libraries, write_entity, write_architecture_header, \
    write_component, write_signal_definitions, write_clock_process, write_uut, write_test_process_header, \
    write_test_process_end, write_architecture_end
from elasticai.generator.generate_specific_testprocess import write_function_test_process_for_multiple_input_results_in_one_output


def main(path_to_testbench: str = '../testbench/') -> None:
    """
    generates the vhd testbench file in the testbench folder for mac async
    Args:
        path_to_testbench: path where testbench is located, default in ../testbench/
    Returns:
        None
    """
    component_name = "mac_async"
    test_bench_file_name = component_name + "_tb.vhd"
    architecture_name = "arch"

    data_width = 16
    frac_width = 8

    # x, y = sigmoid(data_width, frac_width)
    # Note, the two array below, is generated based on data_width and frac_width
    # excitation signals, as test inputs signal
    inputs = {
        "test_X": 0,
        "test_h_in": 0,
        "test_W0": 0,
        "test_W1": 0,
        "test_b": 255,
    }
    # expected signal, as test reference output signal
    outputs = [255]

    with open(path_to_testbench + test_bench_file_name, 'w') as f:
        f.write(write_libraries(math_lib=True))
        f.write(write_entity(entity_name=component_name))
        f.write(write_architecture_header(architecture_name=architecture_name, component_name=component_name))
        f.write(write_component(component_name=component_name, data_width=data_width, frac_width=frac_width,
                                variables_dict={"x1": "in", "x2": "in", "w1": "in", "w2": "in", "b": "in", "y": "out"}))
        f.write(write_signal_definitions(signal_dict={
            "test_X": "signed(16-1 downto 0)",
            "test_h_in": "signed(16-1 downto 0)",
            "test_W0": "signed(16-1 downto 0)",
            "test_W1": "signed(16-1 downto 0)",
            "test_b": "signed(16-1 downto 0)",
            "test_mac_out": "signed(16-1 downto 0)",
            "clk_period": "time := 1 ns",
            "product_1, product_2": "signed(16-1 downto 0)",
        }))
        f.write(write_clock_process())
        f.write(write_uut(component_name=component_name,
                          mapping_dict={
                              "x1": "test_X",
                              "x2": "test_h_in",
                              "w1": "test_W0",
                              "w2": "test_W1",
                              "b": "test_b",
                              "y": "test_mac_out"}))
        f.write(write_test_process_header())
        f.write(write_function_test_process_for_multiple_input_results_in_one_output(inputs_with_names=inputs, outputs=outputs, output_name="test_mac_out"))
        f.write(write_test_process_end())
        f.write(write_architecture_end(architecture_name=architecture_name))


if __name__ == '__main__':
    main()
