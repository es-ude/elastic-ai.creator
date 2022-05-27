import os
from argparse import ArgumentParser

import numpy as np

from elasticai.creator.vhdl.precomputed_scalar_function import Tanh
from elasticai.creator.vhdl.vhdl_formatter.vhdl_formatter import format_vhdl


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--path",
        help="path to folder for generated vhd files",
        required=True,
    )
    args = arg_parser.parse_args()
    if not os.path.isdir(args.path):
        os.mkdir(args.path)
    component_name = "tanh"
    file_path = os.path.join(args.path, f"{component_name}.vhd")

    tanh = Tanh(
        component_name=component_name,
        data_width=16,
        frac_width=8,
        x=np.linspace(-5, 5, 259),
    )
    with open(file_path, "w") as writer:
        tanh_code = tanh()
        for line in tanh_code:
            writer.write(line + "\n")

    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
