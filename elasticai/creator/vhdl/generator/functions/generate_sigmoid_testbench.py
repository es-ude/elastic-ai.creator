from elasticai.creator.vhdl.generator.precomputed_scalar_function import (
    PrecomputedScalarTestBench,
)
from elasticai.creator.vhdl.generator.generator_functions import get_file_path_string
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl


def main() -> None:
    file_path = get_file_path_string(
        relative_path_from_project_root="vhd_files/testbench",
        file_name="sigmoid_tb.vhd",
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

    # indent all lines of the file
    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
