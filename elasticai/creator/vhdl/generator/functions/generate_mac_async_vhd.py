from elasticai.creator.vhdl.generator.mac_async import MacAsync
from elasticai.creator.vhdl.generator.vhd_strings import (
    get_file_path_string,
)


def main(file_path, component_name, data_width, frac_width):
    with open(file_path, "w") as writer:
        code = MacAsync(component_name, data_width, frac_width)
        for line in code():
            writer.write(line)
            if line[-1] != "\n":
                writer.write("\n")


if __name__ == "__main__":
    component_name = "mac_async"
    data_width = 16
    frac_width = 8
    # specify the file path
    file_path = get_file_path_string(folder_names=["..", "source"], file_name=component_name + ".vhd")
    # generate the vhdl-file
    main(file_path=file_path, component_name=component_name, data_width=data_width, frac_width=frac_width)
