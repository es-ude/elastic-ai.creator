from elasticai.creator.vhdl.generator.lstm_testbench_generator import (
    LSTMCellTestBench,
)
from elasticai.creator.vhdl.generator.generator_functions import get_file_path_string
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl


def main() -> None:
    file_path = get_file_path_string(
        relative_path_from_project_root="vhd_files/testbench",
        file_name="lstm_cell_tb.vhd",
    )

    with open(file_path, "w") as writer:
        lstm_cell = LSTMCellTestBench(
            data_width=16,
            frac_width=8,
            input_size=5,
            hidden_size=20,
            test_x_h_data='x"018a",x"ffb5",x"fdd3",x"0091",x"feeb",x"0099",x"fe72",x"ffa9",x"01da",x"ffc9",x"ff42",x"0090",x"0042",x"ffd4",x"ff53",x"00f0",x"007d",x"0134",x"0015",x"fecd",x"ffff",x"ff7c",x"ffb2",x"fe6c",x"01b4",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000"',
            test_c_data='x"0034",x"ff8d",x"ff6e",x"ff72",x"fee0",x"ffaf",x"fee9",x"ffeb",x"ffe9",x"00af",x"ff2a",x"0000",x"ff40",x"002f",x"009f",x"00a3",x"ffc2",x"024d",x"fe1f",x"fff4",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000"',
            h_out=[
                34,
                -80,
                -32,
                -28,
                -88,
                11,
                -60,
                6,
                -16,
                18,
                -32,
                46,
                -77,
                15,
                70,
                27,
                13,
                112,
                -156,
                3,
            ],
            component_name="lstm_common_gate",
        )
        lstm_cell_code = lstm_cell()
        for line in lstm_cell_code:
            writer.write(line + "\n")

    # indent all lines of the file
    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
