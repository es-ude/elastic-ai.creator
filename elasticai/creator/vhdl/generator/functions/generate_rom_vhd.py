import math
from io import StringIO
from itertools import chain

import numpy as np

from elasticai.creator.vhdl.generator.generator_functions import float_array_to_string
from elasticai.creator.vhdl.generator.vhd_strings import (
    get_file_path_string
)
from elasticai.creator.vhdl.language import (
    ContextClause,
    LibraryClause,
    UseClause,
    Entity,
    InterfaceVariable,
    DataType,
    Mode,
    Architecture,
    Process
)

############################
frac_bits = 4
nbits = 12
Bi = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])  # biases for the input gate
############################

component_name = "rom_bi"
file_name = component_name + ".vhd"
architecture_name = "rom_bi_rtl"

DATA_WIDTH = nbits
ADDR_WIDTH = math.ceil(math.log2(len(Bi)))
ROM_STRING = float_array_to_string(float_array=Bi, frac_bits=frac_bits, nbits=nbits)


def main():
    file_path = get_file_path_string(
        folder_names=["..", "source"],
        file_name=file_name
    )
    with open(file_path, "w") as writer:
        stringio = StringIO("")
        code = build_rom(stringio)
        writer.write(code)


def build_rom(writer: StringIO, ):
    # library
    library = ContextClause(
        library_clause=LibraryClause(logical_name_list=["ieee"]),
        use_clause=UseClause(
            selected_names=[
                "ieee.std_logic_1164.all",
                "ieee.std_logic_unsigned.all",
            ]
        ),
    )
    # entity
    entity = Entity(identifier=component_name)
    entity.port_list.append(
        InterfaceVariable(identifier="clk",
                          identifier_type=DataType.STD_LOGIC,
                          mode=Mode.IN)
    )
    entity.port_list.append(
        InterfaceVariable(identifier="en",
                          identifier_type=DataType.STD_LOGIC,
                          mode=Mode.IN)
    )
    entity.port_list.append(
        InterfaceVariable(identifier="addr",
                          identifier_type=DataType.STD_LOGIC_VECTOR,
                          mode=Mode.IN,
                          range=f"{ADDR_WIDTH}-1 downto 0".format(ADDR_WIDTH=ADDR_WIDTH))
    )
    entity.port_list.append(
        InterfaceVariable(identifier="data",
                          identifier_type=DataType.STD_LOGIC_VECTOR,
                          mode=Mode.OUT,
                          range=f"{DATA_WIDTH}-1 downto 0".format(DATA_WIDTH=DATA_WIDTH))
    )
    # architecture
    architecture = Architecture(
        identifier=architecture_name,
        design_unit=component_name)

    architecture.architecture_declaration_list.append(
        "type rom_bi_array_t is array (0 to 2**{ADDR_WIDTH}-1) of std_logic_vector({DATA_WIDTH}-1 downto 0)".format(
            ADDR_WIDTH=ADDR_WIDTH,
            DATA_WIDTH=DATA_WIDTH)
    )
    architecture.architecture_declaration_list.append(
        "signal ROM : rom_bi_array_t:=({ROM_STRING})".format(ROM_STRING=ROM_STRING)
    )
    architecture.architecture_declaration_list.append(
        "attribute rom_style : string"
    )
    architecture.architecture_declaration_list.append(
        'attribute rom_style of ROM : signal is "block"'
    )

    # define process
    rom_process = Process(identifier="ROM", input_name="clk")
    rom_process.process_statements_list.append(
        "if rising_edge(clk) then \nif (en = '1') then\ndata <= ROM(conv_integer(addr))"
    )
    rom_process.process_statements_list.append("end if")
    rom_process.process_statements_list.append("end if")

    # add process to the architecture
    architecture.architecture_statement_part = rom_process

    code = chain(chain(library(), entity()), architecture())
    for line in code:
        writer.write(line)
        writer.write("\n")

    code = writer.getvalue()
    return code


if __name__ == "__main__":
    main()
