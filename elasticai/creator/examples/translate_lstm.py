import argparse
from functools import partial
from pathlib import Path

import torch.nn

from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers import LSTMTranslationArguments
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
        self.lstm_1 = torch.nn.LSTM(input_size=10, hidden_size=100)
        self.lstm_2 = torch.nn.LSTM(input_size=100, hidden_size=10)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lstm_2(self.lstm_1(x)[0])


def main() -> None:
    args = read_commandline_args()

    model = LSTMModel()

    translated = translator.translate_model(
        model=model, build_function_mapping=DEFAULT_BUILD_FUNCTION_MAPPING
    )
    code_repr = translator.generate_code(
        translatable_layers=translated,
        translation_args=dict(
            AbstractLSTM=LSTMTranslationArguments(
                fixed_point_factory=partial(FixedPoint, total_bits=16, frac_bits=8),
                sigmoid_resolution=(-2.5, 2.5, 256),
                tanh_resolution=(-1, 1, 256),
            )
        ),
    )
    translator.save_code(code_repr=code_repr, path=args.build_dir)


if __name__ == "__main__":
    main()
