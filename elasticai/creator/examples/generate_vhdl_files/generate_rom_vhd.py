from argparse import ArgumentParser

import numpy as np

from elasticai.creator.vhdl.generator.rom import Rom
from elasticai.creator.vhdl.number_representations import (
    FloatToSignedFixedPointConverter,
)
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

    rom_name = "bi_rom"
    data_width = 12
    frac_width = 4
    # biases for the input gate
    Bi = np.array([1.1])

    floats_to_signed_fixed_point_converter = FloatToSignedFixedPointConverter(
        bits_used_for_fraction=frac_width, strict=False
    )
    array_value = [floats_to_signed_fixed_point_converter(x) for x in Bi]

    with open(file_path, "w") as writer:
        rom = Rom(
            rom_name=rom_name,
            data_width=data_width,
            values=array_value,
            resource_option="auto",
        )
        for line in rom():
            writer.write(line)
            if line[-1] != "\n":
                writer.write("\n")

    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
