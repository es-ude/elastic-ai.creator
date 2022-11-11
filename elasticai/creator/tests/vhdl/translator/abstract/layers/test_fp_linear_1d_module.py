import unittest

from elasticai.creator.vhdl.components import FPLinear1dComponent, RomComponent
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers import (
    FPLinear1dModule,
    FPLinear1dTranslationArgs,
)


class FPLinear1dModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.linear = FPLinear1dModule(
            weight=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], bias=[1.0, 2.0]
        )
        self.translation_args = FPLinear1dTranslationArgs(
            fixed_point_factory=FixedPoint.get_factory(total_bits=8, frac_bits=4)
        )

    def test_contains_all_needed_components(self) -> None:
        vhdl_components = self.linear.components(self.translation_args)

        target_components = [
            (FPLinear1dComponent, "fp_linear_1d.vhd"),
            (RomComponent, "w_rom.vhd"),
            (RomComponent, "b_rom.vhd"),
        ]
        actual_components = [(type(x), x.file_name) for x in vhdl_components]

        self.assertEqual(actual_components, target_components)
