from elasticai.creator.vhdl.generator.precomputed_scalar_function import (
    PrecomputedScalarTestBench,
)
from elasticai.creator.vhdl.generator.generator_functions import get_file_path_string


def main() -> None:
    file_path = get_file_path_string(
        folder_names=["..", "testbench"], file_name="sigmoid_tb.vhd"
    )

    with open(file_path, "w") as writer:
        sigmoid = PrecomputedScalarTestBench(
            component_name="sigmoid",
            data_width=16,
            frac_width=8,
            x_list_for_testing=[-1281, -1000, -500],
            y_list_for_testing=[0, 4, 28],
        )
        sigmoid_code = sigmoid()
        for line in sigmoid_code:
            writer.write(line + "\n")


if __name__ == "__main__":
    main()
