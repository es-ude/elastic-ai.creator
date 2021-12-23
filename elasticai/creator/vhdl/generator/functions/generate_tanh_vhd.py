import numpy as np
from elasticai.creator.vhdl.generator.general_strings import *
from elasticai.creator.vhdl.generator.generator_functions import tanh_process

component_name = "tanh"
file_name = component_name + ".vhd"
architecture_name = "tanh_rtl"

DATA_WIDTH = 16
FRAC_WIDTH = 8

# generate 259 evenly spaced numbers between (-5,5)
x_list = np.linspace(-5, 5, 259)


def main():
    file_path = get_file_path_string(folder_names=["..", "source"], file_name=file_name)

    with open(file_path, "w") as writer:
        writer.write(get_libraries_string())
        writer.write(
            get_entity_or_component_string(
                entity_or_component="entity",
                entity_or_component_name=component_name,
                data_width=DATA_WIDTH,
                frac_width=FRAC_WIDTH,
                variables_dict={
                    "x": "in signed(DATA_WIDTH-1 downto 0)",
                    "y": "out signed(DATA_WIDTH-1 downto 0)",
                },
            )
        )
        writer.write(
            get_architecture_header_string(
                architecture_name=architecture_name, component_name=component_name
            )
        )
        writer.write(
            get_process_string(
                component_name=component_name,
                lookup_table_generator_function=tanh_process(x_list),
            )
        )
        writer.write(get_architecture_end_string(architecture_name=architecture_name))


if __name__ == "__main__":
    main()
