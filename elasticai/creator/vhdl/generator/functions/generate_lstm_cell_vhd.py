from elasticai.creator.vhdl.generator.lstm_cell import LstmCell
from elasticai.creator.vhdl.generator.vhd_strings import (
    get_file_path_string,
)


def main(component_name: str, data_width: int, frac_width: int) -> None:
    with open(file_path, "w") as writer:
        code = LstmCell(component_name, data_width, frac_width)
        for line in code():
            writer.write(line)
            if line[-1] != "\n":
                writer.write("\n")


if __name__ == "__main__":
    file_path = get_file_path_string(
        folder_names=["..", "source"], file_name="lstm_cell.vhd"
    )
    component_name = "lstm_cell"
    DATA_WIDTH = 16
    FRAC_WIDTH = 8
    # generate the vhdl file
    main(component_name=component_name, data_width=DATA_WIDTH, frac_width=FRAC_WIDTH)
