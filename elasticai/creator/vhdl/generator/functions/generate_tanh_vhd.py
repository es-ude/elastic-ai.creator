from functools import partial

import numpy as np

from elasticai.creator.vhdl.generator.general_strings import (
    get_libraries_string,
    get_entity_or_component_string,
    get_architecture_header_string,
    get_architecture_end_string,
)
from elasticai.creator.vhdl.generator.generator_functions import tanh_process
from elasticai.creator.vhdl.generator.vhd_strings import (
    get_file_path_string,
    get_process_string,
)


class Tanh:
    def __init__(self, data_width, frac_width, x, component_name="tanh"):
        self.component_name = component_name
        self.data_width = data_width
        self.frac_width = frac_width
        self.x = x

    @property
    def file_name(self) -> str:
        return f"{self.component_name}.vhd"

    @property
    def architecture_name(self) -> str:
        return f"{self.component_name}_rtl"

    def build(self) -> str:
        code = ""
        string_builders = [
            get_libraries_string,
            partial(
                get_entity_or_component_string,
                entity_or_component="entity",
                entity_or_component_name=self.component_name,
                data_width=self.data_width,
                frac_width=self.frac_width,
                variables_dict={
                    "x": "in signed(DATA_WIDTH-1 downto 0)",
                    "y": "out signed(DATA_WIDTH-1 downto 0)",
                },
            ),
            partial(
                get_architecture_header_string,
                architecture_name=self.architecture_name,
                component_name=self.component_name,
            ),
            partial(
                get_process_string,
                component_name=self.component_name,
                lookup_table_generator_function=tanh_process(self.x),
            ),
            partial(
                get_architecture_end_string, architecture_name=self.architecture_name
            ),
        ]
        for function in string_builders:
            code += function()
        return code


def main():
    file_path = get_file_path_string(
        folder_names=["..", "source"], file_name="tanh.vhd"
    )
    tanh = Tanh(data_width=16, frac_width=8, x=np.linspace(-5, 5, 259))
    with open(file_path, "w") as writer:
        writer.write(tanh.build())


if __name__ == "__main__":
    main()
