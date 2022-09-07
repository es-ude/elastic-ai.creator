import argparse
from pathlib import Path

import torch

from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers import (
    Linear1dTranslationArgs,
    LSTMTranslationArgs,
)
from elasticai.creator.vhdl.translator.pytorch import translator
from elasticai.creator.vhdl.translator.pytorch.build_function_mappings import (
    DEFAULT_BUILD_FUNCTION_MAPPING,
)


def read_commandline_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_dir", required=True, type=Path)
    return parser.parse_args()


class LSTMModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=10)
        self.linear = torch.nn.Linear(in_features=10, out_features=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.linear(self.lstm(x)[0])


def main() -> None:
    args = read_commandline_args()

    model = LSTMModel()

    fixed_point_factory = FixedPoint.get_factory(total_bits=8, frac_bits=4)
    work_library_name = "xil_defaultlib"
    translation_args = dict(
        LSTMModule=LSTMTranslationArgs(
            fixed_point_factory=fixed_point_factory,
            sigmoid_resolution=(-2.5, 2.5, 256),
            tanh_resolution=(-1, 1, 256),
            work_library_name=work_library_name,
        ),
        Linear1dModule=Linear1dTranslationArgs(
            fixed_point_factory=fixed_point_factory,
            work_library_name=work_library_name,
        ),
    )

    vhdl_modules = translator.translate_model(
        model=model, build_function_mapping=DEFAULT_BUILD_FUNCTION_MAPPING
    )

    code_repr = translator.generate_code(
        vhdl_modules=vhdl_modules, translation_args=translation_args
    )

    translator.save_code(code_repr=code_repr, path=args.build_dir)


if __name__ == "__main__":
    main()
