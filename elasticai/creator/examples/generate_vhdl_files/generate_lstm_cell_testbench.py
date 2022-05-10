from argparse import ArgumentParser

from elasticai.creator.vhdl.generator.lstm_testbench_generator import LSTMCellTestBench
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl


def main() -> None:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--file",
        help="filepath of the generated vhd file",
        required=True,
    )
    args = arg_parser.parse_args()
    file_path = args.file

    with open(file_path, "w") as writer:
        lstm_cell = LSTMCellTestBench(
            data_width=16,
            frac_width=8,
            input_size=5,
            hidden_size=20,
            test_x_h_data=[
                "018a",
                "ffb5",
                "fdd3",
                "0091",
                "feeb",
                "0099",
                "fe72",
                "ffa9",
                "01da",
                "ffc9",
                "ff42",
                "0090",
                "0042",
                "ffd4",
                "ff53",
                "00f0",
                "007d",
                "0134",
                "0015",
                "fecd",
                "ffff",
                "ff7c",
                "ffb2",
                "fe6c",
                "01b4",
                "0000",
                "0000",
                "0000",
                "0000",
                "0000",
                "0000",
                "0000",
            ],
            test_c_data=[
                "0034",
                "ff8d",
                "ff6e",
                "ff72",
                "fee0",
                "ffaf",
                "fee9",
                "ffeb",
                "ffe9",
                "00af",
                "ff2a",
                "0000",
                "ff40",
                "002f",
                "009f",
                "00a3",
                "ffc2",
                "024d",
                "fe1f",
                "fff4",
                "0000",
                "0000",
                "0000",
                "0000",
                "0000",
                "0000",
                "0000",
                "0000",
                "0000",
                "0000",
                "0000",
                "0000",
            ],
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

    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
