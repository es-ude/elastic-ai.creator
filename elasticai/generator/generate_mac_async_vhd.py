from elasticai.generator.generator_writer import *
from elasticai.generator.string_definitions import *

component_name = "mac_async"
file_name = component_name + ".vhd"
architecture_name = "mac_async_rtl"

DATA_WIDTH = 16
DATA_FRAC = 8


def main():
    with open(get_path_file("source", "generated_" + file_name), "w") as writer:
        writer.write(get_libraries_string())
        writer.write(write_entity(entity_name=component_name, data_width=DATA_WIDTH, frac_width=DATA_FRAC,
                                  variables_dict={"x1": "in", "x2": "in", "w1": "in", "w2": "in", "b": "in",
                                                  "y": "out"}))
        writer.write(get_architecture_header_string(architecture_name=architecture_name, component_name=component_name))
        writer.write(write_architecture_signals_definition(["product_1", "product_2"]))
        writer.write(get_begin_architecture_string())
        writer.write(write_mac_async_architecture_behavior())
        writer.write(get_architecture_end_string(architecture_name=architecture_name))


if __name__ == '__main__':
    main()
