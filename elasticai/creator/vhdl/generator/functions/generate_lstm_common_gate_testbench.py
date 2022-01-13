from elasticai.creator.vhdl.generator.testbench_strings import write_testbench_file


def main(path_to_testbench: str = "../../testbench/") -> None:
    """
    generates the vhd testbench file in the testbench folder for lstm common gate
    Args:
        path_to_testbench: path where testbench is located, default in ../testbench/
    Returns:
        None
    """
    component_name = "lstm_common_gate"
    test_bench_file_name = component_name + "_tb.vhd"
    architecture_name = "arch"

    data_width = 16
    frac_width = 8
    vector_len_width = 4

    components_variables_dict = {
        "reset": "in std_logic",
        "clk": "in std_logic",
        "x": "in signed(DATA_WIDTH-1 downto 0)",
        "w": "in signed(DATA_WIDTH-1 downto 0)",
        "b": "in signed(DATA_WIDTH-1 downto 0)",
        "vector_len": "in unsigned(VECTOR_LEN_WIDTH-1 downto 0)",
        "idx": "out unsigned(VECTOR_LEN_WIDTH-1 downto 0)",
        "ready": "out std_logic",
        "y": "out signed(DATA_WIDTH-1 downto 0)",
    }
    type_definitions_dict = {
        "RAM_ARRAY": "array (0 to 9 ) of signed(DATA_WIDTH-1 downto 0)"
    }
    signal_definition_dict = {
        "clk_period": "time := 2 ps",
        "clock": "std_logic",
        "reset, ready": "std_logic:='0'",
        "X_MEM": "RAM_ARRAY :=(others=>(others=>'0'))",
        "W_MEM": "RAM_ARRAY:=(others=>(others=>'0'))",
        "x, w, y, b": "signed(DATA_WIDTH-1 downto 0):=(others=>'0')",
        "vector_len": "unsigned(VECTOR_LEN_WIDTH-1 downto 0):=(others=>'0')",
        "idx": "unsigned(VECTOR_LEN_WIDTH-1 downto 0):=(others=>'0')",
    }
    uut_mapping_dict = {
        "reset": "reset",
        "clk": "clock",
        "x": "x",
        "w": "w",
        "b": "b",
        "vector_len": "vector_len",
        "idx": "idx",
        "ready": "ready",
        "y": "y",
    }
    variable_definitions_before_test_process_dict = {
        "x": "X_MEM(to_integer(idx))",
        "w": "W_MEM(to_integer(idx))",
    }
    variable_definitions_in_test_process_dict = {
        "    vector_len": "to_unsigned(10, VECTOR_LEN_WIDTH)"
    }
    # Note, the two array below, is generated based on data_width and frac_width
    # excitation signals, as test inputs signal
    inputs = [
        {
            "X_MEM": '(x"0013",x"0000",x"0010",x"0013",x"000c",x"0005",x"0005",x"0013",x"0004",x"0002")',
            "W_MEM": '(x"0011",x"0018",x"0000",x"000d",x"0014",x"000f",x"0012",x"0007",x"0017",x"0012")',
            "b": 'x"008a"',
        },
        {
            "X_MEM": '(x"0014",x"000d",x"0017",x"0008",x"0002",x"0007",x"0002",x"0015",x"0001",x"0010")',
            "W_MEM": '(x"000e",x"0014",x"0005",x"0015",x"0009",x"0013",x"0007",x"0016",x"0008",x"0004")',
            "b": 'x"0064"',
        },
        {
            "X_MEM": '(x"000f",x"0017",x"000d",x"000f",x"0001",x"0009",x"0002",x"0007",x"0008",x"0013")',
            "W_MEM": '(x"0001",x"000a",x"0008",x"0010",x"0008",x"0001",x"0016",x"0013",x"0016",x"000a")',
            "b": 'x"009b"',
        },
        {
            "X_MEM": '(x"000c",x"0007",x"0001",x"0019",x"0008",x"000c",x"0019",x"000b",x"0008",x"000d")',
            "W_MEM": '(x"000e",x"0015",x"0001",x"000b",x"0014",x"0012",x"000f",x"0000",x"0008",x"000e")',
            "b": 'x"004c"',
        },
        {
            "X_MEM": '(x"0005",x"0013",x"0002",x"0013",x"000c",x"000f",x"0003",x"0004",x"0010",x"0001")',
            "W_MEM": '(x"0006",x"000d",x"0005",x"0009",x"0017",x"0017",x"000e",x"000d",x"0000",x"0019")',
            "b": 'x"0092"',
        },
    ]
    # expected signal, as test reference output signal
    outputs = [142, 105, 159, 82, 150]

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
        output_name_for_testcases="y",
        math_lib=True,
        vector_len_width=vector_len_width,
        type_definitions_dict=type_definitions_dict,
        clock_name="clock",
        variable_definitions_before_test_process_dict=variable_definitions_before_test_process_dict,
        variable_definitions_in_test_process_dict=variable_definitions_in_test_process_dict,
    )


if __name__ == "__main__":
    main()
