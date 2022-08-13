import os
from argparse import ArgumentParser

from elasticai.creator.vhdl.components.rom_component import RomComponent
from elasticai.creator.vhdl.number_representations import float_values_to_fixed_point
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--path",
        help="path to folder for generated vhd files",
        required=True,
    )
    args = arg_parser.parse_args()
    if not os.path.isdir(args.path):
        os.mkdir(args.path)
    rom_name = "bi_rom"
    file_path = os.path.join(args.path, f"{rom_name}.vhd")

    data_width = 12
    frac_width = 4
    # biases for the input gate
    bi = [1.1]

    with open(file_path, "w") as writer:
        rom = RomComponent(
            rom_name=rom_name,
            values=float_values_to_fixed_point(
                bi, total_bits=data_width, frac_bits=frac_width
            ),
            resource_option="auto",
        )
        for line in rom():
            writer.write(line)
            if line[-1] != "\n":
                writer.write("\n")

    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
