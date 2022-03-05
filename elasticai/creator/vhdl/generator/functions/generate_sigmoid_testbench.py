from elasticai.creator.vhdl.generator.precomputed_scalar_function import (
    SigmoidTestBench,
)
from elasticai.creator.vhdl.generator.vhd_strings import get_file_path_string


def main() -> None:
    file_path = get_file_path_string(
        folder_names=["..", "testbench"], file_name="sigmoid_tb.vhd"
    )

    with open(file_path, "w") as writer:
        s = SigmoidTestBench(component_name="sigmoid_tb", data_width=16, frac_width=8)
        writer.write(s.build())


if __name__ == "__main__":
    main()
