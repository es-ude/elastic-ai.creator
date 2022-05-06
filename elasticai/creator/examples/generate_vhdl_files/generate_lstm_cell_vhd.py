from argparse import ArgumentParser

from elasticai.creator.vhdl.generator.lstm_cell import LstmCell
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

    component_name = "lstm_cell"
    data_width = 16
    frac_width = 8

    # generate the vhdl file
    with open(file_path, "w") as writer:
        code = LstmCell(component_name, data_width, frac_width)
        for line in code():
            writer.write(line)
            if line[-1] != "\n":
                writer.write("\n")

    # indent all lines of the file
    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()

