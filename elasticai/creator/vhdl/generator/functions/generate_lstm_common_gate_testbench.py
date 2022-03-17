from elasticai.creator.vhdl.generator.lstm_testbench_generator import (
    LSTMCommonGateTestBench,
)
from elasticai.creator.vhdl.generator.generator_functions import get_file_path_string
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl


def main() -> None:
    file_path = get_file_path_string(
        folder_names=["..", "testbench"], file_name="lstm_common_gate_tb.vhd"
    )

    with open(file_path, "w") as writer:
        lstm_common_gate = LSTMCommonGateTestBench(
            data_width=16,
            frac_width=8,
            vector_len_width=4,
            component_name="lstm_common_gate",
            x_mem_list_for_testing=[
                'x"0013",x"0000",x"0010",x"0013",x"000c",x"0005",x"0005",x"0013",x"0004",x"0002"',
                'x"0014",x"000d",x"0017",x"0008",x"0002",x"0007",x"0002",x"0015",x"0001",x"0010"',
                'x"000f",x"0017",x"000d",x"000f",x"0001",x"0009",x"0002",x"0007",x"0008",x"0013"',
                'x"000c",x"0007",x"0001",x"0019",x"0008",x"000c",x"0019",x"000b",x"0008",x"000d"',
                'x"0005",x"0013",x"0002",x"0013",x"000c",x"000f",x"0003",x"0004",x"0010",x"0001"',
            ],
            w_mem_list_for_testing=[
                'x"0011",x"0018",x"0000",x"000d",x"0014",x"000f",x"0012",x"0007",x"0017",x"0012"',
                'x"000e",x"0014",x"0005",x"0015",x"0009",x"0013",x"0007",x"0016",x"0008",x"0004"',
                'x"0001",x"000a",x"0008",x"0010",x"0008",x"0001",x"0016",x"0013",x"0016",x"000a"',
                'x"000e",x"0015",x"0001",x"000b",x"0014",x"0012",x"000f",x"0000",x"0008",x"000e"',
                'x"0006",x"000d",x"0005",x"0009",x"0017",x"0017",x"000e",x"000d",x"0000",x"0019"',
            ],
            b_list_for_testing=['x"008a"', 'x"0064"', 'x"009b"', 'x"004c"', 'x"0092"'],
            y_list_for_testing=[142, 105, 159, 82, 150],
        )
        lstm_common_gate_code = lstm_common_gate()
        for line in lstm_common_gate_code:
            writer.write(line + "\n")

    # indent all lines of the file
    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
