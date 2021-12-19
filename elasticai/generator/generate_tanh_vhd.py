import numpy as np
from elasticai.generator.generator_writer import *
from elasticai.generator.generator_functions import tanh_process

component_name = "tanh"
file_name = component_name + ".vhd"
architecture_name = "tanh_rtl"

DATA_WIDTH = 16
FRAC_WIDTH = 8

# generate 259 evenly spaced numbers between (-5,5)
x_list = np.linspace(-5, 5, 259)


def main():
    with open(get_path_file("source", "generated_" + file_name), "w") as writer:
        writer.write(write_libraries())
        writer.write(write_entity(entity_name=component_name, data_width=DATA_WIDTH, frac_width=FRAC_WIDTH,
                                  variables_dict={"x": "in", "y": "out"}))
        writer.write(write_architecture_header(architecture_name=architecture_name, component_name=component_name))
        writer.write(write_process(component_name=component_name, process_name=tanh_process(x_list)))
        writer.write(write_architecture_end(architecture_name=architecture_name))


if __name__ == '__main__':
    main()
