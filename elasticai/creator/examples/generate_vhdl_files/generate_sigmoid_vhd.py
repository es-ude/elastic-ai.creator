import numpy as np

from elasticai.creator.vhdl.generator.precomputed_scalar_function import Sigmoid
from elasticai.creator.vhdl.generator.generator_functions import get_file_path_string
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl


def main():
    file_path = get_file_path_string(
        relative_path_from_project_root="vhdl_resources/source",
        file_name="sigmoid.vhd",
    )

    with open(file_path, "w") as writer:
        sigmoid = Sigmoid(data_width=16, frac_width=8, x=np.linspace(-5, 5, 66))
        sigmoid_code = sigmoid()
        for line in sigmoid_code:
            writer.write(line + "\n")

    # indent all lines of the file
    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
