import unittest

from elasticai.creator.vhdl.code_files.fp_linear_1d_component import FPLinear1dFile
from elasticai.creator.vhdl.code_files.rom_component import RomFile
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers import FPLinear1dModule


class FPLinear1dModuleTest(unittest.TestCase):
    def test_contains_all_needed_components(self) -> None:
        linear = FPLinear1dModule(
            layer_id="ll1",
            weight=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            bias=[1.0, 2.0],
            fixed_point_factory=FixedPoint.get_factory(total_bits=8, frac_bits=4),
        )
        vhdl_components = linear.files

        target_components = [
            (
                FPLinear1dFile,
                "fp_linear_1d_{layer_name}.vhd".format(layer_name=linear.layer_id),
            ),
            (
                RomFile,
                "w_rom_fp_linear_1d_{layer_name}.vhd".format(
                    layer_name=linear.layer_id
                ),
            ),
            (
                RomFile,
                "b_rom_fp_linear_1d_{layer_name}.vhd".format(
                    layer_name=linear.layer_id
                ),
            ),
        ]
        actual_components = [(type(x), x.name) for x in vhdl_components]

        self.assertEqual(target_components, actual_components)
