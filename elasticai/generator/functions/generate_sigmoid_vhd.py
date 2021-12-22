import numpy as np

from elasticai.generator.vhd_strings import *
from elasticai.generator.general_strings import *

from elasticai.generator.generator_functions import sigmoid_process

component_name = "sigmoid"
file_name = component_name + ".vhd"
architecture_name = "sigmoid_rtl"

DATA_WIDTH = 16
DATA_FRAC = 8

# generate 66 evenly spaced numbers between (-5,5)
x_list = np.linspace(-5, 5, 66)  # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]


def main():
    file_path = get_file_path_string(folder_names=["..", "source"],
                                     file_name="generated_" + file_name)

    with open(file_path, "w") as writer:
        writer.write(get_libraries_string())
        writer.write(get_entity_or_component_string(
            entity_or_component="entity",
            entity_or_component_name=component_name,
            data_width=DATA_WIDTH,
            frac_width=DATA_FRAC,
            variables_dict={
                "x": "in signed(DATA_WIDTH-1 downto 0)",
                "y": "out signed(DATA_WIDTH-1 downto 0)"}))
        writer.write(get_architecture_header_string(architecture_name=architecture_name, component_name=component_name))
        writer.write(get_process_string(component_name=component_name, lookup_table_generator_function=sigmoid_process(x_list)))
        writer.write(get_architecture_end_string(architecture_name=architecture_name))


if __name__ == '__main__':
    main()
