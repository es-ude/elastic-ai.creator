import unittest
from dataclasses import dataclass
from typing import Any

import elasticai.creator.qat.layers as qtorch
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.translator.build_function_mapping import (
    BuildFunctionMapping,
)
from elasticai.creator.vhdl.vhdl_files import VHDLFile, VHDLModule


class MockVHDLFile(VHDLFile):
    @property
    def name(self) -> str:
        return "test_component"

    def code(self) -> Code:
        return ["line1", "line2"]


@dataclass
class MockModule(VHDLModule):
    vhdl_components: list[VHDLFile]

    def files(self, args: Any) -> list[VHDLFile]:
        return self.vhdl_components


def mock_build_function1(layer: Any) -> VHDLModule:
    return MockModule([MockVHDLFile()])


def mock_build_function2(layer: Any) -> VHDLModule:
    return MockModule([MockVHDLFile()])


def mock_build_function3(layer: Any) -> VHDLModule:
    return MockModule([MockVHDLFile()])


class BuildFunctionMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.linear_layer = qtorch.QLinear(1, 1, qtorch.Binarize())
        self.conv1d_layer = qtorch.QConv1d(1, 1, (1,), qtorch.Binarize())
        self.conv2d_layer = qtorch.QConv2d(1, 1, (1, 1), qtorch.Binarize())

        self.mapping_dict = {
            "torch.nn.utils.parametrize.ParametrizedQLinear": mock_build_function1,
            "torch.nn.utils.parametrize.ParametrizedQConv1d": mock_build_function2,
        }

        self.mapping = BuildFunctionMapping(mapping=self.mapping_dict)

    def test_get_from_layer_existing_key(self) -> None:
        self.assertEqual(
            mock_build_function1, self.mapping.get_from_layer(self.linear_layer)
        )

    def test_get_from_layer_not_existing_key(self) -> None:
        self.assertEqual(None, self.mapping.get_from_layer(self.conv2d_layer))

    def test_to_dict(self) -> None:
        self.assertEqual(self.mapping_dict, self.mapping.to_dict())

    def test_join_with_dict_new_and_existing_keys(self) -> None:
        new_mapping = self.mapping.join_with_dict(
            {
                "torch.nn.utils.parametrize.ParametrizedQConv2d": mock_build_function3,
                "torch.nn.utils.parametrize.ParametrizedQConv1d": mock_build_function1,
            }
        )
        target_mapping_dict = {
            "torch.nn.utils.parametrize.ParametrizedQLinear": mock_build_function1,
            "torch.nn.utils.parametrize.ParametrizedQConv1d": mock_build_function1,
            "torch.nn.utils.parametrize.ParametrizedQConv2d": mock_build_function3,
        }
        self.assertEqual(target_mapping_dict, new_mapping.to_dict())
