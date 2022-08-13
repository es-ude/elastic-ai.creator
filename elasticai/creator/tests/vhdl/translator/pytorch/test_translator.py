import unittest
from dataclasses import dataclass
from typing import Any, Iterable

import torch.nn

from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.translator.abstract.layers import AbstractLSTM
from elasticai.creator.vhdl.translator.build_function_mapping import (
    BuildFunctionMapping,
)
from elasticai.creator.vhdl.translator.pytorch import translator
from elasticai.creator.vhdl.translator.pytorch.build_function_mappings import (
    DEFAULT_BUILD_FUNCTION_MAPPING,
)
from elasticai.creator.vhdl.translator.pytorch.translator import CodeFile, Module
from elasticai.creator.vhdl.vhdl_component import VHDLComponent, VHDLModule


@dataclass
class VHDLComponentMock:
    name: str
    code: list[str]

    @property
    def file_name(self) -> str:
        return self.name

    def __call__(self) -> Code:
        yield from self.code


@dataclass
class TranslatableMock:
    components: list[VHDLComponent]

    def translate(self, args: Any) -> VHDLModule:
        yield from self.components


def fake_build_function(module: torch.nn.Module) -> TranslatableMock:
    return TranslatableMock(
        components=[
            VHDLComponentMock(name="component1", code=["1", "2", "3"]),
            VHDLComponentMock(name="component2", code=["4", "5", "6"]),
        ]
    )


def unpack_module_directories(
    modules: Iterable[Module],
) -> list[tuple[str, list[tuple[str, Code]]]]:
    def unpack_code_file(code_file: CodeFile) -> tuple[str, Code]:
        return code_file.file_name, list(code_file.code)

    return [
        (module.module_name, list(map(unpack_code_file, module.files)))
        for module in modules
    ]


class TranslatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.build_mapping = BuildFunctionMapping(
            mapping={"torch.nn.modules.rnn.LSTM": fake_build_function}
        )

    def test_translate_model_empty_model(self) -> None:
        model = torch.nn.Sequential()
        translated_model = translator.translate_model(model, self.build_mapping)
        self.assertEqual(len(list(translated_model)), 0)

    def test_translate_model_with_one_layer(self) -> None:
        model = torch.nn.Sequential(torch.nn.LSTM(input_size=1, hidden_size=2))

        translated_model = list(translator.translate_model(model, self.build_mapping))

        self.assertEqual(len(translated_model), 1)
        self.assertEqual(type(translated_model[0]), TranslatableMock)

    def test_generate_code(self) -> None:
        model = torch.nn.Sequential(torch.nn.LSTM(input_size=1, hidden_size=2))
        translated_model = translator.translate_model(model, self.build_mapping)
        modules = translator.generate_code(
            translatable_layers=translated_model, translation_args=dict()
        )

        code = unpack_module_directories(modules)
        expected_code = [
            (
                "0_TranslatableMock",
                [
                    ("component1", ["1", "2", "3"]),
                    ("component2", ["4", "5", "6"]),
                ],
            )
        ]

        self.assertEqual(code, expected_code)

    @unittest.SkipTest
    def test_translate_model_correct_ordering_of_layers(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm_2 = torch.nn.LSTM(input_size=2, hidden_size=3)
                self.lstm_1 = torch.nn.LSTM(input_size=1, hidden_size=2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lstm_2(self.lstm_1(x))

        def extract_input_hidden_size(lstm: AbstractLSTM) -> tuple[int, int]:
            hidden_size = len(lstm.weights_hh[0][0])
            input_size = len(lstm.weights_ih[0][0])
            return input_size, hidden_size

        model = Model()
        translated = list(
            translator.translate_model(model, DEFAULT_BUILD_FUNCTION_MAPPING)
        )

        self.assertEqual(extract_input_hidden_size(translated[0]), (1, 2))  # type: ignore
        self.assertEqual(extract_input_hidden_size(translated[1]), (2, 3))  # type: ignore
