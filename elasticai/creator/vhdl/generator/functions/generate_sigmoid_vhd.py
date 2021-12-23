import numpy as np

from elasticai.creator.vhdl.generator.general_strings import (
    get_libraries_string,
    get_entity_or_component_string,
    get_architecture_header_string,
    get_architecture_end_string,
)
from elasticai.creator.vhdl.generator.generator_functions import sigmoid_process
from elasticai.creator.vhdl.generator.vhd_strings import (
    get_file_path_string,
    get_process_string,
)

component_name = "sigmoid"
file_name = component_name + ".vhd"
architecture_name = "sigmoid_rtl"

DATA_WIDTH = 16
DATA_FRAC = 8

# generate 66 evenly spaced numbers between (-5,5)
x_list = np.linspace(-5, 5, 66)  # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]


class Sigmoid:
    def __init__(
        self,
        data_width: int,
        data_frac: int,
        x,
        component_name: str = "sigmoid",
    ):
        self.component_name = component_name
        self.data_width = data_width
        self.data_frac = data_frac
        self.x = x

    @property
    def file_name(self) -> str:
        return f"{self.component_name}.vhd"

    @property
    def architecture_name(self) -> str:
        return f"{self.component_name}_rtl"

    def build(self) -> str:
        code = get_libraries_string()
        code += get_entity_or_component_string(
            entity_or_component="entity",
            entity_or_component_name=component_name,
            data_width=self.data_width,
            frac_width=self.data_frac,
            variables_dict={
                "x": "in signed(DATA_WIDTH-1 downto 0)",
                "y": "out signed(DATA_WIDTH-1 downto 0)",
            },
        )
        code += get_architecture_header_string(
            architecture_name=self.architecture_name, component_name=self.component_name
        )
        code += get_process_string(
            component_name=self.component_name,
            lookup_table_generator_function=sigmoid_process(self.x),
        )
        code += get_architecture_end_string(architecture_name=self.architecture_name)


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
                lookup_table_generator_function=sigmoid_process(x_list),
            )
        )
        writer.write(get_architecture_end_string(architecture_name=architecture_name))


if __name__ == "__main__":
    main()
