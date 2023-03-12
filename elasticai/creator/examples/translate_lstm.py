import torch

from elasticai.creator.nn.linear import FixedPointLinear
from elasticai.creator.nn.lstm import FixedPointLSTMWithHardActivations
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl_for_deprecation.translator.pytorch import translator


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        fp_factory = FixedPoint.get_builder(8, 4)

        self.lstm = FixedPointLSTMWithHardActivations(
            input_size=3,
            hidden_size=5,
            bias=True,
            batch_first=True,
            fp_config=FixedPoint.get_builder(8, 4),
        )

        self.linear = FixedPointLinear(
            in_features=5, out_features=1, bias=True, config=fp_factory
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.lstm(x))


def main() -> None:
    model = Model()
    translation_args = {
        "0": {"work_library_name": "work"},
        "1": {"work_library_name": "work"},
    }
    code_repr = translator.translate_model(model, translation_args=translation_args)
    code_modules = list(code_repr)
    translator.save_code(code_modules, "build/")


if __name__ == "__main__":
    main()
