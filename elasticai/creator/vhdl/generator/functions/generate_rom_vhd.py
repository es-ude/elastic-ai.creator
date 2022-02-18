from io import StringIO

from elasticai.creator.vhdl.generator.vhd_strings import (
    get_file_path_string
)
from elasticai.creator.vhdl.language import (
    ContextClause,
    LibraryClause,
    UseClause,
    Entity,
    InterfaceVariable,
    DataType, Mode
)

component_name = "rom_bi"
file_name = component_name + ".vhd"
architecture_name = "rom_bi_rtl"

DATA_WITH = 12
# TODO: chnage this, it should be generated from other param -- > math.ceil(math.log2(len(Bi)))
PARAM_ADDR_WIDTH = 3  # will be calculated !


def main():
    file_path = get_file_path_string(
        folder_names=["..", "source"],
        file_name=file_name
    )
    with open(file_path, "w") as writer:
        stringio = StringIO("")
        code = build_rom(stringio)
        writer.write(code)


def build_rom(writer: StringIO):
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
                          variable_type=DataType.STD_LOGIC,
                          mode=Mode.IN)
    )
    entity.port_list.append(
        InterfaceVariable(identifier="en",
                          variable_type=DataType.STD_LOGIC,
                          mode=Mode.IN)
    )
    entity.port_list.append(
        InterfaceVariable(identifier="addr",
                          variable_type=DataType.STD_LOGIC_VECTOR,
                          mode=Mode.IN,
                          range=f"{PARAM_ADDR_WIDTH}-1 downto 0".format(PARAM_ADDR_WIDTH=PARAM_ADDR_WIDTH))
    )
    entity.port_list.append(
        InterfaceVariable(identifier="data",
                          variable_type=DataType.STD_LOGIC_VECTOR,
                          mode=Mode.OUT,
                          range=f"{PARAM_ADDR_WIDTH}-1 downto 0".format(PARAM_ADDR_WIDTH=PARAM_ADDR_WIDTH))
        ####
    )

    for line in library():
        writer.write(line)
        writer.write("\n")

    for line in entity():
        writer.write(line)
        writer.write("\n")

    code = writer.getvalue()
    return code


if __name__ == "__main__":
    main()
