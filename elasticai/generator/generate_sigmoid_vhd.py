from generator_fun import \
    get_path_file, write_libraries, write_entity,\
    write_architecture_header, write_process, write_architecture_end

component_name = "sigmoid"
file_name = component_name + ".vhd"
architecture_name = "sigmoid_rtl"

DATA_WIDTH = 16
DATA_FRAC = 8


def main():
    with open(get_path_file("source", "generated_" + file_name), "w") as writer:
        writer.write(write_libraries())
        writer.write(write_entity(entity_name=component_name, data_width=DATA_WIDTH, frac_width=DATA_FRAC))
        writer.write(write_architecture_header(architecture_name=architecture_name, component_name=component_name))
        writer.write(write_process(component_name=component_name))
        writer.write(write_architecture_end(architecture_name=architecture_name))


if __name__ == '__main__':
    main()
