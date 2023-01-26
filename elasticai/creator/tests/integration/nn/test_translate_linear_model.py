import unittest

import torch

from elasticai.creator.nn.hard_sigmoid import HardSigmoid
from elasticai.creator.nn.linear import FixedPointLinear
from elasticai.creator.nn.relu import ReLU
from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointFactory
from elasticai.creator.vhdl.translator.pytorch import translator


class FixedPointModel(torch.nn.Module):
    def __init__(self, fixed_point_factory: FixedPointFactory) -> None:
        super().__init__()

        self.linear1 = FixedPointLinear(
            in_features=3,
            out_features=4,
            bias=True,
            fixed_point_factory=fixed_point_factory,
        )
        self.linear2 = FixedPointLinear(
            in_features=4,
            out_features=1,
            bias=True,
            fixed_point_factory=fixed_point_factory,
        )
        self.hard_sigmoid = HardSigmoid()
        self.relu1 = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.hard_sigmoid(x)
        return x


class TranslateLinearModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(22)
        torch.cuda.manual_seed_all(22)

        fixed_point_factory = FixedPoint.get_factory(total_bits=8, frac_bits=4)
        model = FixedPointModel(fixed_point_factory)

        translation_args = {
            "0": {"work_library_name": "work"},
            "1": {"work_library_name": "work"},
            "2": {"fixed_point_factory": fixed_point_factory},
            "3": {"fixed_point_factory": fixed_point_factory},
        }
        code_repr = translator.translate_model(model, translation_args=translation_args)
        self.code_modules = list(code_repr)

    def test_code_structure_matches_expected(self) -> None:
        actual_code_structure = [
            (code_module.name, {file.name for file in code_module.files})
            for code_module in self.code_modules
        ]

        expected_code_structure = [
            (
                "0_FixedPointLinear",
                {
                    "fp_linear_1d_0.vhd",
                    "w_rom_fp_linear_1d_0.vhd",
                    "b_rom_fp_linear_1d_0.vhd",
                },
            ),
            (
                "1_FixedPointLinear",
                {
                    "fp_linear_1d_1.vhd",
                    "w_rom_fp_linear_1d_1.vhd",
                    "b_rom_fp_linear_1d_1.vhd",
                },
            ),
            ("2_HardSigmoid", {"fp_hard_sigmoid.vhd"}),
            ("3_ReLU", {"fp_relu_3.vhd"}),
            ("network_component", {"network.vhd"}),
        ]

        self.assertEqual(len(actual_code_structure), len(expected_code_structure))

        for expected_name, expected_files in expected_code_structure:
            with self.subTest(f"module {expected_name} in actual"):
                self.assertIn((expected_name, expected_files), actual_code_structure)
