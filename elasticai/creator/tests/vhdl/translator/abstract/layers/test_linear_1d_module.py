import unittest

from elasticai.creator.vhdl.components import (
    Linear1dComponent,
    LSTMCommonComponent,
    RomComponent,
)
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers import (
    Linear1dModule,
    Linear1dTranslationArgs,
)


class Linear1dModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.linear = Linear1dModule(weight=[[1, 2, 3]], bias=[1])
        self.translation_args = Linear1dTranslationArgs(
            fixed_point_factory=FixedPoint.get_factory(total_bits=8, frac_bits=4)
        )

    def test_contains_all_needed_components(self) -> None:
        vhdl_components = self.linear.components(self.translation_args)

        target_components = [
            (Linear1dComponent, "linear_1d.vhd"),
            (RomComponent, "w_rom.vhd"),
            (RomComponent, "b_rom.vhd"),
            (LSTMCommonComponent, "lstm_common.vhd"),
        ]
        actual_components = [(type(x), x.file_name) for x in vhdl_components]

        self.assertEqual(actual_components, target_components)
