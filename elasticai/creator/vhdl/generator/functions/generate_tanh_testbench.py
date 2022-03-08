from elasticai.creator.vhdl.generator.precomputed_scalar_function import (
    PrecomputedScalarTestBench,
)
from elasticai.creator.vhdl.generator.generator_functions import get_file_path_string


def main() -> None:
    file_path = get_file_path_string(
        folder_names=["..", "testbench"], file_name="tanh_tb.vhd"
    )

    with open(file_path, "w") as writer:
        tanh = PrecomputedScalarTestBench(
            component_name="tanh",
            data_width=16,
            frac_width=8,
            x_list_for_testing=[-1281, -1000, -500, 0, 500, 800, 1024],
            y_list_for_testing=["1111111100000000", -255, -246, 0, 245, 254, 255],
        )
        tanh_code = tanh()
        for line in tanh_code:
            writer.write(line + "\n")


if __name__ == "__main__":
    main()
