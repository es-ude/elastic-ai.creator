import unittest
from dataclasses import dataclass
from typing import Any

from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.translator.abstract.translatable import Translatable
from elasticai.creator.vhdl.translator.build_function_mapping import (
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


def mock_build_function1(layer: Any) -> Translatable:
    return MockTranslatable([MockVHDLComponent()])


def mock_build_function2(layer: Any) -> Translatable:
    return MockTranslatable([MockVHDLComponent()])


def mock_build_function3(layer: Any) -> Translatable:
    return MockTranslatable([MockVHDLComponent()])


@dataclass
class MockLayer1:
    x = 1


@dataclass
class MockLayer2:
    x = 2


@dataclass
class MockLayer3:
    x = 3


class BuildFunctionMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping_dict = {
            "test_build_function_mapping.MockLayer1": mock_build_function1,
            "test_build_function_mapping.MockLayer2": mock_build_function2,
        }
        self.mapping = BuildFunctionMapping(mapping=self.mapping_dict)

    def test_get_from_layer_existing_key(self) -> None:
        self.assertEqual(mock_build_function1, self.mapping.get_from_layer(MockLayer1))
        self.assertEqual(
            mock_build_function1, self.mapping.get_from_layer(MockLayer1())
        )

    def test_get_from_layer_not_existing_key(self) -> None:
        self.assertEqual(None, self.mapping.get_from_layer(MockLayer3))

    def test_to_dict(self) -> None:
        self.assertEqual(self.mapping_dict, self.mapping.to_dict())

    def test_join_with_dict_new_and_existing_keys(self) -> None:
        new_mapping = self.mapping.join_with_dict(
            {
                "test_build_function_mapping.MockLayer3": mock_build_function3,
                "test_build_function_mapping.MockLayer2": mock_build_function1,
            }
        )
        target_mapping_dict = {
            "test_build_function_mapping.MockLayer1": mock_build_function1,
            "test_build_function_mapping.MockLayer2": mock_build_function1,
            "test_build_function_mapping.MockLayer3": mock_build_function3,
        }
        self.assertEqual(target_mapping_dict, new_mapping.to_dict())
