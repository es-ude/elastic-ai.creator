import sys

import torch

from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointFactory
from elasticai.creator.vhdl.quantized_modules.hard_sigmoid import FixedPointHardSigmoid
from elasticai.creator.vhdl.quantized_modules.linear import FixedPointLinear
from elasticai.creator.vhdl.translator.abstract.layers.fp_hard_sigmoid_module import (
    FPHardSigmoidTranslationArgs,
)
from elasticai.creator.vhdl.translator.abstract.layers.linear_1d_module import (
    Linear1dTranslationArgs,
)
from elasticai.creator.vhdl.translator.build_function_mapping import (
    BuildFunctionMapping,
)
from elasticai.creator.vhdl.translator.pytorch import translator
from elasticai.creator.vhdl.translator.pytorch.build_function_mappings import (
    DEFAULT_BUILD_FUNCTION_MAPPING,
)
from elasticai.creator.vhdl.translator.pytorch.build_functions.fp_hard_sigmoid_build_function import (
    build_fp_hard_sigmoid,
)
from elasticai.creator.vhdl.translator.pytorch.build_functions.linear_1d_build_function import (
    build_linear_1d,
)


class FixedPointModel(torch.nn.Module):
    def __init__(self, fixed_point_factory: FixedPointFactory) -> None:
        super().__init__()

        # self.linear1 = FixedPointLinear(
        #     in_features=2, out_features=3, fixed_point_factory=fixed_point_factory
        # )
        self.linear2 = FixedPointLinear(
            in_features=3, out_features=1, fixed_point_factory=fixed_point_factory
        )
        self.hard_sigmoid = FixedPointHardSigmoid(
            fixed_point_factory=fixed_point_factory
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hard_sigmoid(self.linear2(x))


def get_custom_build_mapping() -> BuildFunctionMapping:
    return DEFAULT_BUILD_FUNCTION_MAPPING.join_with_dict(
        {
            "elasticai.creator.vhdl.quantized_modules.linear.FixedPointLinear": build_linear_1d,
            "elasticai.creator.vhdl.quantized_modules.hard_sigmoid.FixedPointHardSigmoid": build_fp_hard_sigmoid,
        }
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Please supply a build directory path as a program argument.")
        return
    build_path = sys.argv[1]

    fixed_point_factory = FixedPoint.get_factory(total_bits=8, frac_bits=4)

    model = FixedPointModel(fixed_point_factory)

    translation_args = dict(
        FixedPointLinear=Linear1dTranslationArgs(
            fixed_point_factory=fixed_point_factory, work_library_name="work"
        ),
        FixedPointHardSigmoid=FPHardSigmoidTranslationArgs(
            fixed_point_factory=fixed_point_factory
        ),
    )

    code_repr = translator.translate_model(
        model=model,
        translation_args=translation_args,
        build_function_mapping=get_custom_build_mapping(),
    )

    translator.save_code(code_repr=code_repr, path=build_path)


if __name__ == "__main__":
    main()
