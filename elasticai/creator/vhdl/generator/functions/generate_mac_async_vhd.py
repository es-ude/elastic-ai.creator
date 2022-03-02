from io import StringIO

from elasticai.creator.vhdl.generator.vhdl_formatter import (
    format_vhdl
)
from elasticai.creator.vhdl.language import (
    Entity,
    InterfaceVariable,
    DataType,
    Architecture,
    InterfaceConstrained,
    Mode,
    InterfaceSignal,
    ContextClause,
    LibraryClause,
    UseClause,
)
from elasticai.creator.vhdl.generator.vhd_strings import (
    get_file_path_string,
    get_mac_async_architecture_behavior_string,
)

component_name = "mac_async"
file_name = component_name + ".vhd"
architecture_name = "mac_async_rtl"

DATA_WIDTH = 16
DATA_FRAC = 8


def main():
    file_path = get_file_path_string(folder_names=["..", "source"], file_name=file_name)
    with open(file_path, "w") as writer:
        stringio = StringIO("")
        code = build_mac_async(stringio)
        writer.write(code)
    # indent the generated vhdl file
    format_vhdl(file_path=file_path)


def build_mac_async(writer: StringIO):
    lib = ContextClause(
        library_clause=LibraryClause(logical_name_list=["ieee"]),
        use_clause=UseClause(
            selected_names=[
                "ieee.std_logic_1164.all",
                "ieee.numeric_std.all",
            ]
        ),
    )
    for line in lib():
        writer.write(line)
        writer.write("\n")
    entity = Entity(component_name)
    entity.generic_list.append(
        InterfaceVariable(
            identifier="DATA_WIDTH", identifier_type=DataType.INTEGER, value=DATA_WIDTH
        )
    )
    entity.generic_list.append(
        InterfaceVariable(
            identifier="FRAC_WIDTH", identifier_type=DataType.INTEGER, value=DATA_FRAC
        )
    )
    entity.port_list.append(
        InterfaceVariable(
            identifier="x1",
            mode=Mode.IN,
            range="DATA_WIDTH-1 downto 0",
            identifier_type=DataType.SIGNED,
        )
    )
    entity.port_list.append(
        InterfaceVariable(
            identifier="x2",
            mode=Mode.IN,
            range="DATA_WIDTH-1 downto 0",
            identifier_type=DataType.SIGNED,
        )
    )
    entity.port_list.append(
        InterfaceVariable(
            identifier="w1",
            mode=Mode.IN,
            range="DATA_WIDTH-1 downto 0",
            identifier_type=DataType.SIGNED,
        )
    )
    entity.port_list.append(
        InterfaceVariable(
            identifier="w2",
            mode=Mode.IN,
            range="DATA_WIDTH-1 downto 0",
            identifier_type=DataType.SIGNED,
        )
    )
    entity.port_list.append(
        InterfaceVariable(
            identifier="b",
            mode=Mode.IN,
            range="DATA_WIDTH-1 downto 0",
            identifier_type=DataType.SIGNED,
        )
    )
    entity.port_list.append(
        InterfaceVariable(
            identifier="y",
            mode=Mode.OUT,
            range="DATA_WIDTH-1 downto 0",
            identifier_type=DataType.SIGNED,
        )
    )
    for line in entity():
        writer.write(line)
        writer.write("\n")
    architecture = Architecture(
        identifier=architecture_name,
        design_unit=component_name,
    )
    architecture.architecture_statement_part = (
        get_mac_async_architecture_behavior_string
    )
    architecture.architecture_declaration_list.append(
        InterfaceSignal(
            identifier="product_1",
            range="DATA_WIDTH-1 downto 0",
            identifier_type=DataType.SIGNED,
        )
    )
    architecture.architecture_declaration_list.append(
        InterfaceSignal(
            identifier="product_2",
            range="DATA_WIDTH-1 downto 0",
            identifier_type=DataType.SIGNED,
        )
    )
    for line in architecture():
        writer.write(line)
        writer.write("\n")

    code = writer.getvalue()
    return code


if __name__ == "__main__":
    main()
