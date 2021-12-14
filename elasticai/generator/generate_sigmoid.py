from elasticai.generator.generate_testbench import write_libraries, write_entity, write_architecture_header, \
    write_component, write_signal_definitions, write_clock_process, write_uut, write_test_process_header, \
    write_test_process_end, write_architecture_end
from elasticai.generator.generate_specific_testprocess import write_function_test_process


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
        f.write(write_function_test_process(inputs, outputs))
        f.write(write_test_process_end())
        f.write(write_architecture_end(architecture_name=architecture_name))


if __name__ == '__main__':
    main()