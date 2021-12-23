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
            entity_or_component_name=self.component_name,
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
        return code


def main():
    file_path = get_file_path_string(
        folder_names=["..", "source"], file_name="sigmoid.vhd"
    )

    with open(file_path, "w") as writer:
        s = Sigmoid(data_width=16, data_frac=8, x=np.linspace(-5, 5, 66))
        writer.write(s.build())


if __name__ == "__main__":
    main()
