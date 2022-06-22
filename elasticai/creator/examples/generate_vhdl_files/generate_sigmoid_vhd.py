import os
from argparse import ArgumentParser

import numpy as np

from elasticai.creator.vhdl.number_representations import float_values_to_fixed_point
from elasticai.creator.vhdl.precomputed_scalar_function import Sigmoid
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
    component_name = "sigmoid"
    file_path = os.path.join(args.path, f"{component_name}.vhd")

    # noinspection PyTypeChecker
    data = float_values_to_fixed_point(
        np.linspace(-5, 5, 66).tolist(), total_bits=16, frac_bits=8
    )

    sigmoid = Sigmoid(
        component_name=component_name,
        x=data,
    )
    sigmoid_code = sigmoid()

    with open(file_path, "w") as writer:
        for line in sigmoid_code:
            writer.write(line + "\n")

    format_vhdl(file_path=file_path)


if __name__ == "__main__":
    main()
