from argparse import ArgumentParser

from elasticai.creator.vhdl.generator.mac_async import MacAsync
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--file",
        help="filepath of the generated vhd file",
        required=True,
    )
    args = arg_parser.parse_args()
    file_path = args.file

    component_name = "mac_async"
    data_width = 16
    frac_width = 8

    with open(file_path, "w") as writer:
        code = MacAsync(component_name, data_width, frac_width)
        for line in code():
            writer.write(line)
            if line[-1] != "\n":
                writer.write("\n")

    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
