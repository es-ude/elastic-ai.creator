from dataclasses import dataclass
from typing import Iterable
from unittest import TestCase

import torch.nn

from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.translator.pytorch import translator
from elasticai.creator.vhdl.translator.pytorch.build_mapping import BuildMapping
from elasticai.creator.vhdl.translator.pytorch.translator import (
    CodeFile,
    ModuleDirectory,
)
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

    def translate(self, *args, **kwargs) -> VHDLModule:
        yield from self.components


def fake_build_function(module: torch.nn.Module) -> TranslatableMock:
    return TranslatableMock(
        components=[
            VHDLComponentMock(name="component1", code=["1", "2", "3"]),
            VHDLComponentMock(name="component2", code=["4", "5", "6"]),
        ]
    )


def unpack_module_directories(
    modules: Iterable[ModuleDirectory],
) -> list[tuple[str, list[tuple[str, Code]]]]:
    def unpack_code_file(code_file: CodeFile) -> tuple[str, Code]:
        return code_file.file_name, list(code_file.code_lines)

    return [
        (module.dir_name, list(map(unpack_code_file, module.code_files)))
        for module in modules
    ]


class TranslatorTest(TestCase):
    def setUp(self) -> None:
        self.build_mapping = BuildMapping()
        self.build_mapping.set(torch.nn.LSTMCell, fake_build_function)

    def test_translate_model_empty_model(self) -> None:
        model = torch.nn.Sequential()
        translated_model = translator.translate_model(model, self.build_mapping)
        self.assertEqual(len(list(translated_model)), 0)

    def test_translate_model_with_one_layer(self) -> None:
        model = torch.nn.Sequential(torch.nn.LSTMCell(input_size=1, hidden_size=2))

        translated_model = translator.translate_model(model, self.build_mapping)
        translated_model = list(translated_model)

        self.assertEqual(len(translated_model), 1)
        self.assertEqual(type(translated_model[0]), TranslatableMock)

    def test_generate_code(self) -> None:
        model = torch.nn.Sequential(torch.nn.LSTMCell(input_size=1, hidden_size=2))
        translated_model = translator.translate_model(model, self.build_mapping)
        modules = translator.generate_code(
            translatable_model=translated_model, translation_args=dict()
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
