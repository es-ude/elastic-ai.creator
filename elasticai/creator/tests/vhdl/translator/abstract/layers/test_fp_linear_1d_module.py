import unittest

from elasticai.creator.vhdl.components import FPLinear1dFile, RomComponent
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers import (
    FPLinear1dModule,
    FPLinear1dTranslationArgs,
)


class FPLinear1dModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.linear = FPLinear1dModule(
            layer_id="ll1", weight=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], bias=[1.0, 2.0]
        )
        self.translation_args = FPLinear1dTranslationArgs(
            fixed_point_factory=FixedPoint.get_factory(total_bits=8, frac_bits=4)
        )

    def test_contains_all_needed_components(self) -> None:
        vhdl_components = self.linear.files(self.translation_args)

        target_components = [
            (
                FPLinear1dFile,
                "fp_linear_1d_{layer_name}.vhd".format(layer_name=self.linear.layer_id),
            ),
            (
                RomComponent,
                "w_rom_fp_linear_1d_{layer_name}.vhd".format(
                    layer_name=self.linear.layer_id
                ),
            ),
            (
                RomComponent,
                "b_rom_fp_linear_1d_{layer_name}.vhd".format(
                    layer_name=self.linear.layer_id
                ),
            ),
        ]
        actual_components = [(type(x), x.name) for x in vhdl_components]

        self.assertEqual(target_components, actual_components)
