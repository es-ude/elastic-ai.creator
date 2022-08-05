import unittest
from dataclasses import dataclass
from typing import Any

import torch.nn

from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.translator.abstract.translatable import Translatable
from elasticai.creator.vhdl.translator.pytorch.build_function_mapping import (
    BuildFunctionMapping,
)
from elasticai.creator.vhdl.vhdl_component import VHDLComponent, VHDLModule


class MockVHDLComponent(VHDLComponent):
    @property
    def file_name(self) -> str:
        return "test_component"

    def __call__(self) -> Code:
        return ["line1", "line2"]


@dataclass
class MockTranslatable(Translatable):
    components: list[VHDLComponent]

    def translate(self, args: Any) -> VHDLModule:
        return self.components


def mock_build_function1(layer: torch.nn.Module) -> Translatable:
    return MockTranslatable([MockVHDLComponent()])


def mock_build_function2(layer: torch.nn.Module) -> Translatable:
    return MockTranslatable([MockVHDLComponent()])


def mock_build_function3(layer: torch.nn.Module) -> Translatable:
    return MockTranslatable([MockVHDLComponent()])


class BuildFunctionMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping_dict = {
            "torch.nn.modules.linear.Linear": mock_build_function1,
            "torch.nn.modules.conv.Conv1d": mock_build_function2,
        }
        self.mapping = BuildFunctionMapping(mapping=self.mapping_dict)

    def test_get_from_layer_existing_key(self) -> None:
        self.assertEqual(
            self.mapping.get_from_layer(torch.nn.Linear), mock_build_function1
        )
        self.assertEqual(
            self.mapping.get_from_layer(torch.nn.Linear(1, 1)), mock_build_function1
        )

    def test_get_from_layer_not_existing_key(self) -> None:
        self.assertEqual(self.mapping.get_from_layer(torch.nn.LSTM), None)

    def test_to_dict(self) -> None:
        self.assertEqual(self.mapping.to_dict(), self.mapping_dict)

    def test_join_with_dict_new_and_existing_keys(self) -> None:
        new_mapping = self.mapping.join_with_dict(
            {
                "torch.nn.modules.conv.Conv2d": mock_build_function3,
                "torch.nn.modules.conv.Conv1d": mock_build_function1,
            }
        )
        target_mapping_dict = {
            "torch.nn.modules.linear.Linear": mock_build_function1,
            "torch.nn.modules.conv.Conv2d": mock_build_function3,
            "torch.nn.modules.conv.Conv1d": mock_build_function1,
        }
        self.assertEqual(new_mapping.to_dict(), target_mapping_dict)
