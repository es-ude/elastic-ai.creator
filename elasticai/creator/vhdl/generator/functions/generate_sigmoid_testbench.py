from elasticai.creator.vhdl.generator.testbench_strings import write_testbench_file


def main(path_to_testbench: str = "../../testbench/") -> None:
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

    components_variables_dict = {
        "x": "in signed(DATA_WIDTH-1 downto 0)",
        "y": "out signed(DATA_WIDTH-1 downto 0)",
    }
    signal_definition_dict = {
        "clk_period": "time := 1 ns",
        "test_input": "signed(16-1 downto 0):=(others=>'0')",
        "test_output": "signed(16-1 downto 0)",
    }
    uut_mapping_dict = {"x": "test_input", "y": "test_output"}

    # x, y = sigmoid(data_width, frac_width)
    # Note, the two array below, is generated based on data_width and frac_width
    # excitation signals, as test inputs signal
    inputs = [-1281, -1000, -500]
    # expected signal, as test reference output signal
    outputs = [0, 4, 28]

    write_testbench_file(
        path_to_testbench=path_to_testbench,
        test_bench_file_name=test_bench_file_name,
        component_name=component_name,
        architecture_name=architecture_name,
        data_width=data_width,
        frac_width=frac_width,
        component_variables_dict=components_variables_dict,
        signal_definitions_dict=signal_definition_dict,
        uut_mapping_dict=uut_mapping_dict,
        inputs_for_testcases=inputs,
        outputs_for_testcases=outputs,
        input_name_for_testcases="test_input",
        output_name_for_testcases="test_output",
    )


if __name__ == "__main__":
    main()
