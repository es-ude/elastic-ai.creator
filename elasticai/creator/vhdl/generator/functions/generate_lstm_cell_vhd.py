from io import StringIO

from elasticai.creator.vhdl.generator.generator_functions import (
    generate_signal_definitions_for_lstm,
)
from elasticai.creator.vhdl.generator.vhd_strings import (
    get_file_path_string,
)
from elasticai.creator.vhdl.language import (
    LibraryClause,
    UseClause,
    Entity,
    InterfaceVariable,
    DataType,
    Mode,
    ComponentDeclaration,
    Architecture,
    InterfaceSignal,
    Process,
    ContextClause,
    PortMap
)
from elasticai.creator.vhdl.generator.vhdl_formatter import (
    format_vhdl
)

component_name = "lstm_cell"
file_name = component_name + ".vhd"
architecture_name = "lstm_cell_rtl"

DATA_WIDTH = 16
FRAC_WIDTH = 8


def main():
    file_path = get_file_path_string(
        folder_names=["..", "source"], file_name=file_name
    )
    with open(file_path, "w") as writer:
        stringio = StringIO("")
        code = build_lstm_cell(stringio)
        writer.write(code)
    # indent the generated vhdl file
    format_vhdl(file_path=file_path)


def build_lstm_cell(writer: StringIO):
    lib = ContextClause(
        library_clause=LibraryClause(logical_name_list=["ieee", "work"]),
        use_clause=UseClause(
            selected_names=[
                "ieee.std_logic_1164.all",
                "ieee.numeric_std.all",
                "work.all",
            ]
        ),
    )

    entity = Entity(identifier="lstm_cell")
    entity.generic_list.append(
        InterfaceVariable(identifier="DATA_WIDTH",
                          variable_type=DataType.INTEGER,
                          value=DATA_WIDTH)
    )
    entity.generic_list.append(
        InterfaceVariable(identifier="FRAC_WIDTH",
                          variable_type=DataType.INTEGER,
                          value=FRAC_WIDTH)
    )

    entity.port_list.append(
        InterfaceVariable(identifier="x",
                          mode=Mode.IN,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    entity.port_list.append(
        InterfaceVariable(identifier="c_in",
                          mode=Mode.IN,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    entity.port_list.append(
        InterfaceVariable(identifier="h_in",
                          mode=Mode.IN,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    entity.port_list.append(
        InterfaceVariable(identifier="c_out",
                          mode=Mode.OUT,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    entity.port_list.append(
        InterfaceVariable(identifier="h_out",
                          mode=Mode.OUT,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )

    # -------------------------------------------------- #
    # define the architecture
    architecture = Architecture(identifier="lstm_cell_rtl",
                                design_unit="lstm_cell")

    lstm_signal_definitions = generate_signal_definitions_for_lstm(
        data_width=DATA_WIDTH, frac_width=FRAC_WIDTH
    )
    # adding the generated signal declarations first
    for identifier, value in lstm_signal_definitions.items():
        architecture.architecture_declaration_list.append(
            InterfaceSignal(identifier=identifier,
                            variable_type=DataType.SIGNED,
                            range="DATA_WIDTH-1 downto 0",
                            value=value,
                            mode=None)
        )
    # ---------------------------------------------------- #
    # adding the intermediate result
    architecture.architecture_declaration_list.append(
        InterfaceSignal(
            identifier="i_wo_activation",
            variable_type=DataType.SIGNED,
            range="DATA_WIDTH-1 downto 0",
            value="(others=>'0')",
            mode=None)
    )

    architecture.architecture_declaration_list.append(
        InterfaceSignal(identifier="i",
                        variable_type=DataType.SIGNED,
                        range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')",
                        mode=None)
    )
    architecture.architecture_declaration_list.append(
        InterfaceSignal(identifier="f_wo_activation",
                        variable_type=DataType.SIGNED,
                        range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')",
                        mode=None)
    )
    architecture.architecture_declaration_list.append(
        InterfaceSignal(identifier="f",
                        variable_type=DataType.SIGNED,
                        range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')",
                        mode=None)
    )
    architecture.architecture_declaration_list.append(
        InterfaceSignal(identifier="g_wo_activation",
                        variable_type=DataType.SIGNED,
                        range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')",
                        mode=None)
    )
    architecture.architecture_declaration_list.append(
        InterfaceSignal(identifier="g",
                        variable_type=DataType.SIGNED,
                        range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')",
                        mode=None)
    )
    architecture.architecture_declaration_list.append(
        InterfaceSignal(identifier="o_wo_activation",
                        variable_type=DataType.SIGNED,
                        range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')",
                        mode=None)
    )
    architecture.architecture_declaration_list.append(
        InterfaceSignal(identifier="o",
                        variable_type=DataType.SIGNED,
                        range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')",
                        mode=None)
    )
    architecture.architecture_declaration_list.append(
        InterfaceSignal(identifier="c_new",
                        variable_type=DataType.SIGNED,
                        range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')",
                        mode=None)
    )
    architecture.architecture_declaration_list.append(
        InterfaceSignal(identifier="c_new_wo_activation",
                        variable_type=DataType.SIGNED,
                        range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')",
                        mode=None)
    )
    architecture.architecture_declaration_list.append(
        InterfaceSignal(identifier="h_new",
                        variable_type=DataType.SIGNED,
                        range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')",
                        mode=None)
    )
    # define components
    mac_async_component = ComponentDeclaration(identifier="mac_async")
    mac_async_component.generic_list.append(
        InterfaceVariable(identifier="DATA_WIDTH",
                          variable_type=DataType.INTEGER,
                          value="DATA_WIDTH")
    )
    mac_async_component.generic_list.append(
        InterfaceVariable(identifier="FRAC_WIDTH",
                          variable_type=DataType.INTEGER,
                          value="FRAC_WIDTH")
    )
    mac_async_component.port_list.append(
        InterfaceVariable(identifier="x1",
                          mode=Mode.IN,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    mac_async_component.port_list.append(
        InterfaceVariable(identifier="x2",
                          mode=Mode.IN,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    mac_async_component.port_list.append(
        InterfaceVariable(identifier="w1",
                          mode=Mode.IN,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    mac_async_component.port_list.append(
        InterfaceVariable(identifier="w2",
                          mode=Mode.IN,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    mac_async_component.port_list.append(
        InterfaceVariable(identifier="b",
                          mode=Mode.IN,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    mac_async_component.port_list.append(
        InterfaceVariable(identifier="y",
                          mode=Mode.OUT,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )

    sigmoid_component = ComponentDeclaration(identifier="sigmoid")
    sigmoid_component.generic_list.append(
        InterfaceVariable(identifier="DATA_WIDTH",
                          variable_type=DataType.INTEGER,
                          value="DATA_WIDTH")
    )
    sigmoid_component.generic_list.append(
        InterfaceVariable(identifier="FRAC_WIDTH",
                          variable_type=DataType.INTEGER,
                          value="FRAC_WIDTH")
    )
    sigmoid_component.port_list.append(
        InterfaceVariable(identifier="x",
                          mode=Mode.IN,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    sigmoid_component.port_list.append(
        InterfaceVariable(identifier="y",
                          mode=Mode.OUT,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )

    tanh_component = ComponentDeclaration(identifier="tanh")
    tanh_component.generic_list.append(
        InterfaceVariable(identifier="DATA_WIDTH",
                          variable_type=DataType.INTEGER,
                          value="DATA_WIDTH")
    )
    tanh_component.generic_list.append(
        InterfaceVariable(identifier="FRAC_WIDTH",
                          variable_type=DataType.INTEGER,
                          value="FRAC_WIDTH")
    )
    tanh_component.port_list.append(
        InterfaceVariable(identifier="x",
                          mode=Mode.IN,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    tanh_component.port_list.append(
        InterfaceVariable(identifier="y",
                          mode=Mode.OUT,
                          variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0")
    )
    # add component to architecture
    architecture.architecture_component_list.append(mac_async_component)
    architecture.architecture_component_list.append(sigmoid_component)
    architecture.architecture_component_list.append(tanh_component)

    # signal assignment
    architecture.architecture_assignment_list.append(
        "c_out <= c_new_wo_activation"
    )
    architecture.architecture_assignment_list.append(
        "h_out <= h_new"
    )

    # port map
    portmap = PortMap(
        map_name="FORGET_GATE_MAC",
        component_name="mac_async"
    )
    portmap.signal_list.append("x1 => x")
    portmap.signal_list.append("x2 => h_in")
    portmap.signal_list.append("w1 => wif")
    portmap.signal_list.append("w2 => whf")
    portmap.signal_list.append("b => bf")
    portmap.signal_list.append("y => f_wo_activation")
    architecture.architecture_port_map_list.append(portmap)

    portmap = PortMap(
        map_name="FORGET_GATE_SIGMOID",
        component_name="sigmoid"
    )
    portmap.signal_list.append("f_wo_activation")
    portmap.signal_list.append("f")
    architecture.architecture_port_map_list.append(portmap)

    portmap = PortMap(
        map_name="INPUT_GATE_MAC",
        component_name="mac_async"
    )
    portmap.signal_list.append("x1 => x")
    portmap.signal_list.append("x2 => h_in")
    portmap.signal_list.append("w1 => wii")
    portmap.signal_list.append("w2 => whi")
    portmap.signal_list.append("b => bi")
    portmap.signal_list.append("y => i_wo_activation")
    architecture.architecture_port_map_list.append(portmap)

    portmap = PortMap(
        map_name="INPUT_GATE_SIGMOID",
        component_name="sigmoid"
    )
    portmap.signal_list.append("i_wo_activation")
    portmap.signal_list.append("i")
    architecture.architecture_port_map_list.append(portmap)

    portmap = PortMap(
        map_name="CELL_GATE_MAC",
        component_name="mac_async"
    )
    portmap.signal_list.append("x1 => x")
    portmap.signal_list.append("x2 => h_in")
    portmap.signal_list.append("w1 => wig")
    portmap.signal_list.append("w2 => whg")
    portmap.signal_list.append("b => bg")
    portmap.signal_list.append("y => g_wo_activation")
    architecture.architecture_port_map_list.append(portmap)

    portmap = PortMap(
        map_name="CELL_GATE_TANH",
        component_name="tanh"
    )
    portmap.signal_list.append("g_wo_activation")
    portmap.signal_list.append("g")
    architecture.architecture_port_map_list.append(portmap)

    portmap = PortMap(
        map_name="NEW_CELL_STATE_MAC",
        component_name="mac_async"
    )
    portmap.signal_list.append("x1 => f")
    portmap.signal_list.append("x2 => i")
    portmap.signal_list.append("w1 => c_in")
    portmap.signal_list.append("w2 => g")
    portmap.signal_list.append("b => (others=>'0')")
    portmap.signal_list.append("y => c_new_wo_activation")
    architecture.architecture_port_map_list.append(portmap)

    portmap = PortMap(
        map_name="NEW_CELL_STATE_TANH",
        component_name="tanh"
    )
    portmap.signal_list.append("c_new_wo_activation")
    portmap.signal_list.append("c_new")
    architecture.architecture_port_map_list.append(portmap)

    portmap = PortMap(
        map_name="MAC_ASYNC_4",
        component_name="mac_async"
    )
    portmap.signal_list.append("x1 => x")
    portmap.signal_list.append("x2 => h_in")
    portmap.signal_list.append("w1 => wio")
    portmap.signal_list.append("w2 => who")
    portmap.signal_list.append("b => bo")
    portmap.signal_list.append("y => o_wo_activation")
    architecture.architecture_port_map_list.append(portmap)

    portmap = PortMap(
        map_name="SIGMOID_1",
        component_name="sigmoid"
    )
    portmap.signal_list.append("x => o_wo_activation")
    portmap.signal_list.append("y => o")
    architecture.architecture_port_map_list.append(portmap)

    # process
    process_content = Process(
        identifier="H_OUT",
        input_name="o,c_new",
    )
    process_content.process_statements_list.append(
        "h_new <= shift_right((o*c_new), FRAC_WIDTH)(DATA_WIDTH-1 downto 0)"
    )
    architecture.architecture_port_map_list.append(process_content)

    # write it in StringIO
    for line in lib():
        writer.write(line)
        writer.write("\n")

    for line in entity():
        writer.write(line)
        writer.write("\n")

    for line in architecture():
        writer.write(line)
        writer.write("\n")

    code = writer.getvalue()
    return code


if __name__ == "__main__":
    main()
