from elasticai.creator.vhdl.generator.precomputed_scalar_function import (
    SigmoidTestBench,
)
from elasticai.creator.vhdl.generator.vhd_strings import get_file_path_string
from elasticai.creator.vhdl.generator.vhdl_formatter import format_vhdl


def main() -> None:
    file_path = get_file_path_string(
        folder_names=["..", "testbench"], file_name="sigmoid_tb.vhd"
    )

    with open(file_path, "w") as writer:
        s = SigmoidTestBench(component_name="sigmoid_tb", data_width=16, frac_width=8)
        for line in s():
            writer.write(line)
            if line[-1] != "\n":
                writer.write("\n")
    # indent the generated vhdl file
    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
