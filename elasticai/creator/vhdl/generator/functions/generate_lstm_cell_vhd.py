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

component_name = "lstm_cell"
file_name = component_name + ".vhd"
architecture_name = "lstm_cell_rtl"

DATA_WIDTH = 16
FRAC_WIDTH = 8


def main():
    file_path = get_file_path_string(folder_names=["..", "source"], file_name=file_name)

    with open(file_path, "w") as writer:
        writer.write(get_libraries_string(work_lib=True))
        writer.write(
            get_entity_or_component_string(
                entity_or_component="entity",
                entity_or_component_name=component_name,
                data_width=DATA_WIDTH,
                frac_width=FRAC_WIDTH,
                variables_dict={
                    "x": "in signed(DATA_WIDTH-1 downto 0)",
                    "c_in": "in signed(DATA_WIDTH-1 downto 0)",
                    "h_in": "in signed(DATA_WIDTH-1 downto 0)",
                    "c_out": "out signed(DATA_WIDTH-1 downto 0)",
                    "h_out": "out signed(DATA_WIDTH-1 downto 0)",
                },
            )
        )

        # architecture structure
        writer.write(
            get_architecture_header_string(
                architecture_name=architecture_name, component_name=component_name
            )
        )

        # generate lstm signal definitions
        lstm_signal_definitions = generate_signal_definitions_for_lstm(
            data_width=DATA_WIDTH, frac_width=FRAC_WIDTH
        )
        writer.write(get_signal_definitions_string(lstm_signal_definitions))

        # string of input/forget/cell/output/ gate definition or new cell state definition
        writer.write(
            get_gate_definition_string(
                comment="-- Intermediate results\n"
                "-- Input gate without/with activation\n"
                "-- i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi})",
                signal_names=["i_wo_activation", "i"],
            )
        )
        writer.write(
            get_gate_definition_string(
                comment="-- Forget gate without/with activation\n"
                "-- f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf})",
                signal_names=["f_wo_activation", "f"],
            )
        )
        writer.write(
            get_gate_definition_string(
                comment="-- Cell gate without/with activation\n"
                "-- g = \\tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg})",
                signal_names=["g_wo_activation", "g"],
            )
        )
        writer.write(
            get_gate_definition_string(
                comment="-- Output gate without/with activation\n"
                "-- o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho})",
                signal_names=["o_wo_activation", "o"],
            )
        )
        writer.write(
            get_gate_definition_string(
                comment="-- new_cell_state without/with activation\n"
                "-- c' = f * c + i * g",
                signal_names=["c_new", "c_new_wo_activation"],
            )
        )
        writer.write(
            get_gate_definition_string(
                comment="""-- h' = o * \\tanh(c')""", signal_names=["h_new"]
            )
        )
        # components
        writer.write(
            get_entity_or_component_string(
                entity_or_component="component",
                entity_or_component_name="mac_async",
                data_width="DATA_WIDTH",
                frac_width="FRAC_WIDTH",
                variables_dict={
                    "x1": "in signed(DATA_WIDTH-1 downto 0)",
                    "x2": "in signed(DATA_WIDTH-1 downto 0)",
                    "w1": "in signed(DATA_WIDTH-1 downto 0)",
                    "w2": "in signed(DATA_WIDTH-1 downto 0)",
                    "b": "in signed(DATA_WIDTH-1 downto 0)",
                    "y": "out signed(DATA_WIDTH-1 downto 0)",
                },
            )
        )
        writer.write(
            get_entity_or_component_string(
                entity_or_component="component",
                entity_or_component_name="sigmoid",
                data_width="DATA_WIDTH",
                frac_width="FRAC_WIDTH",
                variables_dict={
                    "x": "in signed(DATA_WIDTH-1 downto 0)",
                    "y": "out signed(DATA_WIDTH-1 downto 0)",
                },
            )
        )
        writer.write(
            get_entity_or_component_string(
                entity_or_component="component",
                entity_or_component_name="tanh",
                data_width="DATA_WIDTH",
                frac_width="FRAC_WIDTH",
                variables_dict={
                    "x": "in signed(DATA_WIDTH-1 downto 0)",
                    "y": "out signed(DATA_WIDTH-1 downto 0)",
                },
            )
        )
        # architecture behavior
        writer.write(get_architecture_begin_string())
        writer.write(
            get_variable_definitions_string(
                {"c_out": "c_new_wo_activation", "h_out": "h_new"}
            )
        )
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


if __name__ == "__main__":
    main()
