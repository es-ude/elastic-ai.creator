from elasticai.creator.vhdl.generator.general_strings import (
    get_libraries_string,
    get_entity_or_component_string,
    get_architecture_header_string,
    get_signal_definitions_string,
    get_architecture_begin_string,
    get_architecture_end_string,
)
from elasticai.creator.vhdl.generator.vhd_strings import (
    get_file_path_string,
    get_mac_async_architecture_behavior_string,
)

component_name = "mac_async"
file_name = component_name + ".vhd"
architecture_name = "mac_async_rtl"

DATA_WIDTH = 16
DATA_FRAC = 8


def main():
    file_path = get_file_path_string(folder_names=["..", "source"], file_name=file_name)

    with open(file_path, "w") as writer:
        writer.write(get_libraries_string())
        writer.write(
            get_entity_or_component_string(
                entity_or_component="entity",
                entity_or_component_name=component_name,
                data_width=DATA_WIDTH,
                frac_width=DATA_FRAC,
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
            get_architecture_header_string(
                architecture_name=architecture_name, component_name=component_name
            )
        )
        writer.write(
            get_signal_definitions_string(
                signal_dict={
                    "product_1": "signed(DATA_WIDTH-1 downto 0)",
                    "product_2": "signed(DATA_WIDTH-1 downto 0)",
                }
            )
        )
        writer.write(get_architecture_begin_string())
        writer.write(get_mac_async_architecture_behavior_string())
        writer.write(get_architecture_end_string(architecture_name=architecture_name))


if __name__ == "__main__":
    main()
