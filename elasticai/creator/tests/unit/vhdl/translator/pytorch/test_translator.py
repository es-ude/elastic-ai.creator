import unittest
from typing import Iterable

import torch

from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers.fp_linear_1d_module import (
    FPLinear1dTranslationArgs,
)
from elasticai.creator.vhdl.translator.abstract.layers.lstm_module import (
    LSTMTranslationArgs,
)
from elasticai.creator.vhdl.translator.build_function_mapping import (
    BuildFunctionMapping,
)
from elasticai.creator.vhdl.translator.pytorch import translator
from elasticai.creator.vhdl.translator.pytorch.build_function_mappings import (
    DEFAULT_BUILD_FUNCTION_MAPPING,
)
from elasticai.creator.vhdl.code import Code, CodeFile, CodeModule, CodeModuleBase, CodeFileBase


def fake_build_function(module: torch.nn.Module, layer_id: str) -> CodeModuleBase:
    return CodeModuleBase(
        name="module0",
        files=[
            CodeFileBase(name="component1", code=["1", "2", "3"]),
            CodeFileBase(name="component2", code=["4", "5", "6"]),
        ],
    )


def unpack_module_directories(
    modules: Iterable[CodeModule],
) -> list[tuple[str, list[tuple[str, Code]]]]:
    def unpack_code_file(code_file: CodeFile) -> tuple[str, Code]:
        return code_file.name, list(code_file.code())

    return [
        (module.name, list(map(unpack_code_file, module.files))) for module in modules
    ]


class TranslatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.build_mapping = BuildFunctionMapping(
            mapping={"torch.nn.modules.module.Module": fake_build_function}
        )

    def test_translate_model_empty_model(self) -> None:
        model = torch.nn.Sequential()
        generated_code = translator.translate_model(
            model, translation_args=dict(), build_function_mapping=self.build_mapping
        )
        self.assertEqual(len(list(generated_code)), 1)

    def test_translation_yields_one_build_mapping_result_for_first_layer(self) -> None:
        model = torch.nn.Sequential(torch.nn.Module())
        code_containers = translator.translate_model(
            model, translation_args=dict(), build_function_mapping=self.build_mapping
        )

        code = unpack_module_directories(code_containers)
        expected_code = (
            "0_Module",
            [
                ("component1", ["1", "2", "3"]),
                ("component2", ["4", "5", "6"]),
            ],
        )

        self.assertEqual(code[0], expected_code)

    @unittest.SkipTest
    def test_translate_model_correct_ordering_of_layers(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size=1, hidden_size=2)
                self.linear = torch.nn.Linear(in_features=2, out_features=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lstm(self.linear(x))

        model = Model()
        translated = list(
            translator.translate_model(
                model,
                translation_args=dict(
                    Linear=FPLinear1dTranslationArgs(
                        fixed_point_factory=FixedPoint.get_factory(8, 4),
                        work_library_name="work",
                    ),
                    LSTM=LSTMTranslationArgs(
                        fixed_point_factory=FixedPoint.get_factory(8, 4),
                        sigmoid_resolution=(-2.5, 2.5, 100),
                        tanh_resolution=(-2.5, 2.5, 100),
                    ),
                ),
                build_function_mapping=DEFAULT_BUILD_FUNCTION_MAPPING,
            )
        )

        self.assertEqual(translated[0].name, "0_Linear")
        self.assertEqual(translated[1].name, "1_LSTM")
