import numpy as np

from elasticai.creator.vhdl.generator.precomputed_scalar_function import Sigmoid
from elasticai.creator.vhdl.generator.vhd_strings import (
    get_file_path_string,
)


def main():
    file_path = get_file_path_string(
        folder_names=["..", "source"], file_name="sigmoid.vhd"
    )

    with open(file_path, "w") as writer:
        s = Sigmoid(data_width=16, frac_width=8, x=np.linspace(-5, 5, 66))
        writer.write(s.build())


if __name__ == "__main__":
    main()
