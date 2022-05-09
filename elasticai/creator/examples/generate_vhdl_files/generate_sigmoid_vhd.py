from argparse import ArgumentParser

import numpy as np

from elasticai.creator.vhdl.generator.precomputed_scalar_function import Sigmoid
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--file",
        help="filepath of the generated vhd file",
        required=True,
    )
    args = arg_parser.parse_args()
    file_path = args.file

    with open(file_path, "w") as writer:
        sigmoid = Sigmoid(data_width=16, frac_width=8, x=np.linspace(-5, 5, 66))
        sigmoid_code = sigmoid()
        for line in sigmoid_code:
            writer.write(line + "\n")

    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
