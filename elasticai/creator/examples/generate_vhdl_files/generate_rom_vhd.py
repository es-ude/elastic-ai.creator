import math
import numpy as np

from elasticai.creator.vhdl.generator.generator_functions import (
    float_array_to_hex_string,
    get_file_path_string,
)
from elasticai.creator.vhdl.generator.rom import Rom
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl


def main(rom_name, data_width, addr_width, array_value):
    file_path = get_file_path_string(
        relative_path_from_project_root="vhd_files/source",
        file_name=rom_name + ".vhd",
    )
    with open(file_path, "w") as writer:
        rom = Rom(
            rom_name=rom_name,
            data_width=data_width,
            addr_width=addr_width,
            array_value=array_value,
        )
        for line in rom():
            writer.write(line)
            if line[-1] != "\n":
                writer.write("\n")

    # indent all lines of the file
    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    rom_name = "rom_bi"
    data_width = 12
    frac_bits = 4
    # biases for the input gate
    Bi = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    addr_width = math.ceil(math.log2(len(Bi)))
    array_value = float_array_to_hex_string(
        float_array=Bi, frac_bits=frac_bits, nbits=data_width
    )
    # generate the vhdl file
    main(rom_name, data_width, addr_width, array_value)
