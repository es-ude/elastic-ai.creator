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
    Library, Entity, InterfaceVariable, DataType, Mode, ComponentDeclaration, Architecture, InterfaceSignal,
    InterfaceList, Process
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
    # TODO: replace this with CodeGenerator
    writer.write(get_libraries_string(work_lib=True))

    ##############################
    # library = Library()
    # library.more_libs_list = ["work.all"]
    # for line in library():
    #     writer.write(line)
    #     writer.write("\n")
    ################################
    entity = Entity(identifier="lstm_cell")
    entity.generic_list.append(
        InterfaceVariable(identifier="DATA_WIDTH", variable_type=DataType.INTEGER, value=DATA_WIDTH))
    entity.generic_list.append(
        InterfaceVariable(identifier="FRAC_WIDTH", variable_type=DataType.INTEGER, value=FRAC_WIDTH))

    entity.port_list.append(
        InterfaceVariable(identifier="x", mode=Mode.IN, variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0"))
    entity.port_list.append(
        InterfaceVariable(identifier="c_in", mode=Mode.IN, variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0"))
    entity.port_list.append(
        InterfaceVariable(identifier="h_in", mode=Mode.IN, variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0"))
    entity.port_list.append(
        InterfaceVariable(identifier="c_out", mode=Mode.OUT, variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0"))
    entity.port_list.append(
        InterfaceVariable(identifier="h_out", mode=Mode.OUT, variable_type=DataType.SIGNED,
                          range="DATA_WIDTH-1 downto 0"))

    for line in entity():
        writer.write(line)
        writer.write("\n")
    # -------------------------------------------------- #
    # TODO: refactor this !!
    lstm_signal_definitions = generate_signal_definitions_for_lstm(
        data_width=DATA_WIDTH, frac_width=FRAC_WIDTH
    )
    signals_list = InterfaceList()
    # adding the generated signal declarations first
    for identifier, value in lstm_signal_definitions.items():
        signals_list.append(
            InterfaceSignal(identifier=identifier, variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                            value=value, mode=None))
    # ---------------------------------------------------- #
    # adding the intermediate result
    signals_list.append(
        InterfaceSignal(identifier="i_wo_activation", variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')", mode=None))
    signals_list.append(
        InterfaceSignal(identifier="i", variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')", mode=None))
    signals_list.append(
        InterfaceSignal(identifier="f_wo_activation", variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')", mode=None))
    signals_list.append(
        InterfaceSignal(identifier="f", variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')", mode=None))
    signals_list.append(
        InterfaceSignal(identifier="g_wo_activation", variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')", mode=None))
    signals_list.append(
        InterfaceSignal(identifier="g", variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')", mode=None))
    signals_list.append(
        InterfaceSignal(identifier="o_wo_activation", variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')", mode=None))
    signals_list.append(
        InterfaceSignal(identifier="o", variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')", mode=None))
    signals_list.append(
        InterfaceSignal(identifier="c_new", variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')", mode=None))
    signals_list.append(
        InterfaceSignal(identifier="c_new_wo_activation", variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')", mode=None))
    signals_list.append(
        InterfaceSignal(identifier="h_new", variable_type=DataType.SIGNED, range="DATA_WIDTH-1 downto 0",
                        value="(others=>'0')", mode=None))

    # components
    component_list = InterfaceList()

    mac_async_component = ComponentDeclaration(identifier="mac_async")
    mac_async_component.generic_list.append(
        InterfaceVariable(identifier="DATA_WIDTH", variable_type=DataType.INTEGER, value="DATA_WIDTH"))
    mac_async_component.generic_list.append(
        InterfaceVariable(identifier="FRAC_WIDTH", variable_type=DataType.INTEGER, value="FRAC_WIDTH"))
    mac_async_component.port_list.append(InterfaceVariable(identifier="x1", mode=Mode.IN, variable_type=DataType.SIGNED,
                                                           range="DATA_WIDTH-1 downto 0"))
    mac_async_component.port_list.append(InterfaceVariable(identifier="x2", mode=Mode.IN, variable_type=DataType.SIGNED,
                                                           range="DATA_WIDTH-1 downto 0"))
    mac_async_component.port_list.append(InterfaceVariable(identifier="w1", mode=Mode.IN, variable_type=DataType.SIGNED,
                                                           range="DATA_WIDTH-1 downto 0"))
    mac_async_component.port_list.append(InterfaceVariable(identifier="w2", mode=Mode.IN, variable_type=DataType.SIGNED,
                                                           range="DATA_WIDTH-1 downto 0"))
    mac_async_component.port_list.append(InterfaceVariable(identifier="b", mode=Mode.IN, variable_type=DataType.SIGNED,
                                                           range="DATA_WIDTH-1 downto 0"))
    mac_async_component.port_list.append(InterfaceVariable(identifier="y", mode=Mode.OUT, variable_type=DataType.SIGNED,
                                                           range="DATA_WIDTH-1 downto 0"))

    sigmoid_component = ComponentDeclaration(identifier="sigmoid")
    sigmoid_component.generic_list.append(
        InterfaceVariable(identifier="DATA_WIDTH", variable_type=DataType.INTEGER, value="DATA_WIDTH"))
    sigmoid_component.generic_list.append(
        InterfaceVariable(identifier="FRAC_WIDTH", variable_type=DataType.INTEGER, value="FRAC_WIDTH"))
    sigmoid_component.port_list.append(InterfaceVariable(identifier="x", mode=Mode.IN, variable_type=DataType.SIGNED,
                                                         range="DATA_WIDTH-1 downto 0"))
    sigmoid_component.port_list.append(InterfaceVariable(identifier="y", mode=Mode.OUT, variable_type=DataType.SIGNED,
                                                         range="DATA_WIDTH-1 downto 0"))

    tanh_component = ComponentDeclaration(identifier="tanh")
    tanh_component.generic_list.append(
        InterfaceVariable(identifier="DATA_WIDTH", variable_type=DataType.INTEGER, value="DATA_WIDTH"))
    tanh_component.generic_list.append(
        InterfaceVariable(identifier="FRAC_WIDTH", variable_type=DataType.INTEGER, value="FRAC_WIDTH"))
    tanh_component.port_list.append(InterfaceVariable(identifier="x", mode=Mode.IN, variable_type=DataType.SIGNED,
                                                      range="DATA_WIDTH-1 downto 0"))
    tanh_component.port_list.append(InterfaceVariable(identifier="y", mode=Mode.OUT, variable_type=DataType.SIGNED,
                                                      range="DATA_WIDTH-1 downto 0"))
    component_list.append(mac_async_component)
    component_list.append(sigmoid_component)
    component_list.append(tanh_component)
    # port map
    # TODO: implement a port map !!!
    # process
    process_content = Process(identifier="H_OUT_PROCESS", input="o,c_new", lookup_table_generator_function="")
    process_content.sequential_statements_list.append(
        "h_new <= shift_right((o*c_new), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);")

    # architecture structure
    architecture = Architecture(identifier="lstm_cell_rtl",
                                signal_list=signals_list,
                                component_list=component_list,
                                process_content=str(process_content()),
                                design_unit="lstm_cell")

    # for line in architecture():
    #     writer.write(line)
    #     writer.write("\n")

    # writer.write(
    #     get_architecture_header_string(
    #         architecture_name=architecture_name, component_name=component_name
    #     )
    # )

    # string of input/forget/cell/output/ gate definition or new cell state definition
    # writer.write(
    #     get_gate_definition_string(
    #         comment="-- Intermediate results\n"
    #                 "-- Input gate without/with activation\n"
    #                 "-- i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi})",
    #         signal_names=["i_wo_activation", "i"],
    #     )
    # )
    # writer.write(
    #     get_gate_definition_string(
    #         comment="-- Forget gate without/with activation\n"
    #                 "-- f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf})",
    #         signal_names=["f_wo_activation", "f"],
    #     )
    # )
    # writer.write(
    #     get_gate_definition_string(
    #         comment="-- Cell gate without/with activation\n"
    #                 "-- g = \\tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg})",
    #         signal_names=["g_wo_activation", "g"],
    #     )
    # )
    # writer.write(
    #     get_gate_definition_string(
    #         comment="-- Output gate without/with activation\n"
    #                 "-- o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho})",
    #         signal_names=["o_wo_activation", "o"],
    #     )
    # )
    # writer.write(
    #     get_gate_definition_string(
    #         comment="-- new_cell_state without/with activation\n"
    #                 "-- c' = f * c + i * g",
    #         signal_names=["c_new", "c_new_wo_activation"],
    #     )
    # )
    # writer.write(
    #     get_gate_definition_string(
    #         comment="""-- h' = o * \\tanh(c')""", signal_names=["h_new"]
    #     )
    # )
    # components
    # writer.write(
    #     get_entity_or_component_string(
    #         entity_or_component="component",
    #         entity_or_component_name="mac_async",
    #         data_width="DATA_WIDTH",
    #         frac_width="FRAC_WIDTH",
    #         variables_dict={
    #             "x1": "in signed(DATA_WIDTH-1 downto 0)",
    #             "x2": "in signed(DATA_WIDTH-1 downto 0)",
    #             "w1": "in signed(DATA_WIDTH-1 downto 0)",
    #             "w2": "in signed(DATA_WIDTH-1 downto 0)",
    #             "b": "in signed(DATA_WIDTH-1 downto 0)",
    #             "y": "out signed(DATA_WIDTH-1 downto 0)",
    #         },
    #     )
    # )
    # writer.write(
    #     get_entity_or_component_string(
    #         entity_or_component="component",
    #         entity_or_component_name="sigmoid",
    #         data_width="DATA_WIDTH",
    #         frac_width="FRAC_WIDTH",
    #         variables_dict={
    #             "x": "in signed(DATA_WIDTH-1 downto 0)",
    #             "y": "out signed(DATA_WIDTH-1 downto 0)",
    #         },
    #     )
    # )
    # writer.write(
    #     get_entity_or_component_string(
    #         entity_or_component="component",
    #         entity_or_component_name="tanh",
    #         data_width="DATA_WIDTH",
    #         frac_width="FRAC_WIDTH",
    #         variables_dict={
    #             "x": "in signed(DATA_WIDTH-1 downto 0)",
    #             "y": "out signed(DATA_WIDTH-1 downto 0)",
    #         },
    #     )
    # )
    # architecture behavior
    # writer.write(get_architecture_begin_string())
    # writer.write(
    #     get_variable_definitions_string(
    #         {"c_out": "c_new_wo_activation", "h_out": "h_new"}
    #     )
    # )

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
