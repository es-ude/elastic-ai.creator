from elasticai.generator.vhd_strings import *
from elasticai.generator.general_strings import *

component_name = "lstm_cell"
file_name = component_name + ".vhd"
architecture_name = "lstm_cell_rtl"

DATA_WIDTH = 16
FRAC_WIDTH = 8


def main():
    with open(get_path_file("source", "generated_" + file_name), "w") as writer:
        writer.write(get_libraries_string(work_lib=True))
        writer.write(write_entity(entity_name=component_name, data_width=DATA_WIDTH, frac_width=FRAC_WIDTH,
                                  variables_dict={"x": "in", "c_in": "in", "h_in": "in", "c_out": "out",
                                                  "h_out": "out"}))
        # architecture structure
        writer.write(get_architecture_header_string(architecture_name=architecture_name, component_name=component_name))
        writer.write(write_lstm_signals_definition(data_width=DATA_WIDTH, frac_width=FRAC_WIDTH))
        writer.write(write_input_gate(without_activation="i_wo_activation", with_activation="i"))
        writer.write(write_forget_gate(without_activation="f_wo_activation", with_activation="f"))
        writer.write(write_cell_gate(without_activation="g_wo_activation", with_activation="g"))
        writer.write(write_output_gate(without_activation="o_wo_activation", with_activation="o"))
        writer.write(write_new_cell_state(without_activation="c_new", with_activation="c_new_wo_activation"))
        # components
        writer.write(write_component(component_name="mac_async", data_width="DATA_WIDTH", frac_width="FRAC_WIDTH",
                                     variables_dict={"x1": "in", "x2": "in", "w1": "in", "w2": "in", "b": "in",
                                                     "y": "out"}))
        writer.write(write_component(component_name="sigmoid", data_width="DATA_WIDTH", frac_width="FRAC_WIDTH",
                                     variables_dict={"x": "in", "y": "out"}))
        writer.write(write_component(component_name="tanh", data_width="DATA_WIDTH", frac_width="FRAC_WIDTH",
                                     variables_dict={"x": "in", "y": "out"}))
        # architecture behavior
        writer.write(get_begin_architecture_string())
        writer.write(write_signal_map({"c_out": "c_new_wo_activation", "h_out": "h_new"}))
        writer.write(write_port_map(map_name="FORGET_GATE_MAC", component_name="mac_async",
                                    signals={"x1": "x", "x2": "h_in", "w1": "wif", "w2": "whf", "b": "bf",
                                             "y": "f_wo_activation"}))  #
        writer.write(write_port_map(map_name="FORGET_GATE_SIGMOID", component_name="sigmoid",
                                    signals=["f_wo_activation", "f"]))
        writer.write(write_port_map(map_name="INPUT_GATE_MAC", component_name="mac_async",
                                    signals={"x1": "x", "x2": "h_in", "w1": "wii", "w2": "whi", "b": "bi",
                                             "y": "i_wo_activation"}))
        writer.write(write_port_map(map_name="INPUT_GATE_SIGMOID", component_name="sigmoid",
                                    signals=["i_wo_activation", "i"]))
        writer.write(write_port_map(map_name="CELL_GATE_MAC", component_name="mac_async",
                                    signals={"x1": "x", "x2": "h_in", "w1": "wig", "w2": "whg", "b": "bg",
                                             "y": "g_wo_activation"}))
        writer.write(write_port_map(map_name="CELL_GATE_TANH", component_name="tanh",
                                    signals=["g_wo_activation", "g"]))
        writer.write(write_port_map(map_name="NEW_CELL_STATE_MAC", component_name="mac_async",
                                    signals={"x1": "f", "x2": "i", "w1": "c_in", "w2": "g", "b": "(others=>'0')",
                                             "y": "c_new_wo_activation"}))
        writer.write(write_port_map(map_name="NEW_CELL_STATE_TANH", component_name="tanh",
                                    signals=["c_new_wo_activation", "c_new"]))
        writer.write(write_port_map(map_name="MAC_ASYNC_4", component_name="mac_async",
                                    signals={"x1": "x", "x2": "h_in", "w1": "wio", "w2": "who", "b": "bo",
                                             "y": "o_wo_activation"}))
        writer.write(write_port_map(map_name="SIGMOID_1", component_name="sigmoid",
                                    signals={"x": "o_wo_activation", "y": "o"}))
        writer.write(write_lstm_process())
        writer.write(get_architecture_end_string(architecture_name=architecture_name))


if __name__ == '__main__':
    main()
