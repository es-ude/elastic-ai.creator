from io import StringIO

from elasticai.creator.vhdl.generator.general_strings import (
    get_libraries_string,
    get_entity_or_component_string,
    get_architecture_header_string,
    get_signal_definitions_string,
    get_architecture_begin_string,
    get_variable_definitions_string,
    get_architecture_end_string,
)
from elasticai.creator.vhdl.generator.generator_functions import (
    generate_signal_definitions_for_lstm,
)
from elasticai.creator.vhdl.generator.vhd_strings import (
    get_file_path_string,
    get_gate_definition_string,
    get_port_map_string,
    get_define_process_string,
)
from elasticai.creator.vhdl.language import (
    LibraryClause, UseClause, Entity, InterfaceVariable, DataType, Mode, ComponentDeclaration, Architecture,
    InterfaceSignal,
    InterfaceList, Process, ContextClause
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
    for line in lib():
        writer.write(line)
        writer.write("\n")

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

    for line in entity():
        writer.write(line)
        writer.write("\n")
    # -------------------------------------------------- #
    # define the architecture
    architecture = Architecture(identifier="lstm_cell_rtl",
                                design_unit="lstm_cell")
    # TODO: refactor this with some thing similar to precomputed scalar function!!
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
    # TODO: implement signal assignment

    # port map
    # TODO:  implement port map

    # process
    # TODO : refactor the process in langauge to let it have no loookup_table !
    process_content = Process(identifier="H_OUT_PROCESS", input_name="o,c_new",
                              lookup_table_generator_function=[""])
    process_content.process_statements_list.append(
        "h_new <= shift_right((o*c_new), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);")

    for line in architecture():
        writer.write(line)
        writer.write("\n")

# old ones !!
    # port mapping
    writer.write(
        get_port_map_string(
            map_name="FORGET_GATE_MAC",
            component_name="mac_async",
            signals={
                "x1": "x",
                "x2": "h_in",
                "w1": "wif",
                "w2": "whf",
                "b": "bf",
                "y": "f_wo_activation",
            },
        )
    )
    writer.write(
        get_port_map_string(
            map_name="FORGET_GATE_SIGMOID",
            component_name="sigmoid",
            signals=["f_wo_activation", "f"],
        )
    )
    writer.write(
        get_port_map_string(
            map_name="INPUT_GATE_MAC",
            component_name="mac_async",
            signals={
                "x1": "x",
                "x2": "h_in",
                "w1": "wii",
                "w2": "whi",
                "b": "bi",
                "y": "i_wo_activation",
            },
        )
    )
    writer.write(
        get_port_map_string(
            map_name="INPUT_GATE_SIGMOID",
            component_name="sigmoid",
            signals=["i_wo_activation", "i"],
        )
    )
    writer.write(
        get_port_map_string(
            map_name="CELL_GATE_MAC",
            component_name="mac_async",
            signals={
                "x1": "x",
                "x2": "h_in",
                "w1": "wig",
                "w2": "whg",
                "b": "bg",
                "y": "g_wo_activation",
            },
        )
    )
    writer.write(
        get_port_map_string(
            map_name="CELL_GATE_TANH",
            component_name="tanh",
            signals=["g_wo_activation", "g"],
        )
    )
    writer.write(
        get_port_map_string(
            map_name="NEW_CELL_STATE_MAC",
            component_name="mac_async",
            signals={
                "x1": "f",
                "x2": "i",
                "w1": "c_in",
                "w2": "g",
                "b": "(others=>'0')",
                "y": "c_new_wo_activation",
            },
        )
    )
    writer.write(
        get_port_map_string(
            map_name="NEW_CELL_STATE_TANH",
            component_name="tanh",
            signals=["c_new_wo_activation", "c_new"],
        )
    )
    writer.write(
        get_port_map_string(
            map_name="MAC_ASYNC_4",
            component_name="mac_async",
            signals={
                "x1": "x",
                "x2": "h_in",
                "w1": "wio",
                "w2": "who",
                "b": "bo",
                "y": "o_wo_activation",
            },
        )
    )
    writer.write(
        get_port_map_string(
            map_name="SIGMOID_1",
            component_name="sigmoid",
            signals={"x": "o_wo_activation", "y": "o"},
        )
    )
    # write process
    writer.write(
        get_define_process_string(
            process_name="H_OUT_PROCESS",
            sensitive_signals_list=["o", "c_new"],
            behavior="h_new <= shift_right((o*c_new), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);",
        )
    )
    writer.write(get_architecture_end_string(architecture_name=architecture_name))

    code = writer.getvalue()
    return code


if __name__ == "__main__":
    main()
