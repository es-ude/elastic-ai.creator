import numpy as np

from generator_fun import \
    get_path_file, write_libraries, write_entity, \
    write_architecture_header, write_process, write_architecture_end, sigmoid_process

component_name = "sigmoid"
file_name = component_name + ".vhd"
architecture_name = "sigmoid_rtl"

DATA_WIDTH = 16
DATA_FRAC = 8

# generate 66 evenly spaced numbers between (-5,5)
x_list = np.linspace(-5, 5, 66)  # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]


def main():
    with open(get_path_file("source", "generated_" + file_name), "w") as writer:
        writer.write(write_libraries())
        writer.write(write_entity(entity_name=component_name, data_width=DATA_WIDTH, frac_width=DATA_FRAC))
        writer.write(write_architecture_header(architecture_name=architecture_name, component_name=component_name))
        writer.write(write_process(component_name=component_name, process_name=sigmoid_process(x_list)))
        writer.write(write_architecture_end(architecture_name=architecture_name))


if __name__ == '__main__':
    main()
