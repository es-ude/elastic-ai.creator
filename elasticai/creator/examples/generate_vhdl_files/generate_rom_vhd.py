import math
import numpy as np

from elasticai.creator.vhdl.generator.generator_functions import (
    get_file_path_string,
)
from elasticai.creator.vhdl.generator.rom import (
    Rom,
    pad_with_zeros,
)
from elasticai.creator.vhdl.number_representations import (
    FloatToSignedFixedPointConverter,
    FloatToHexFixedPointStringConverter,
)
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
            resource_option="auto",
        )
        for line in rom():
            writer.write(line)
            if line[-1] != "\n":
                writer.write("\n")

    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    rom_name = "bi_rom"
    data_width = 12
    frac_width = 4
    # biases for the input gate
    Bi = np.array([1.1])
    addr_width = math.ceil(math.log2(len(Bi)))
    if addr_width == 0:
        addr_width = 1

    floats_to_signed_fixed_point_converter = FloatToSignedFixedPointConverter(
        bits_used_for_fraction=frac_width, strict=False
    )
    float_to_hex_fixed_point_string_converter = FloatToHexFixedPointStringConverter(
        total_bit_width=data_width,
        as_signed_fixed_point=floats_to_signed_fixed_point_converter,
    )
    array_value = [float_to_hex_fixed_point_string_converter(x) for x in Bi]
    array_value = pad_with_zeros(array_value)

    main(rom_name, data_width, addr_width, array_value)
