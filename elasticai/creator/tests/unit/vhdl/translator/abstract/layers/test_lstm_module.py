from unittest import TestCase

from vhdl.code_files import (
    LSTMCommonVHDLFile,
)
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers import (
    LSTMModule,
    LSTMTranslationArgs,
)
from vhdl.code_files.dual_port_2_clock_ram_component import DualPort2ClockRamVHDLFile
from vhdl.code_files.lstm_component import LSTMFile
from vhdl.code_files.rom_component import RomFile
from vhdl.code_files.sigmoid_component import SigmoidComponent
from vhdl.code_files.tanh_component import TanhComponent


class LSTMModuleTest(TestCase):
    def setUp(self) -> None:
        self.lstm = LSTMModule(
            weights_ih=[[[1, 2], [3, 4], [5, 6], [7, 8]]],
            weights_hh=[[[1], [2], [3], [4]]],
            biases_ih=[[1, 2, 3, 4]],
            biases_hh=[[5, 6, 7, 8]],
        )

        self.translation_args = LSTMTranslationArgs(
            fixed_point_factory=FixedPoint.get_factory(total_bits=8, frac_bits=2),
            sigmoid_resolution=(-2.5, 2.5, 256),
            tanh_resolution=(-1, 1, 256),
        )

    def test_correct_number_of_components(self) -> None:
        vhdl_components = list(self.lstm.files(self.translation_args))
        self.assertEqual(len(vhdl_components), 13)

    def test_contains_all_needed_components(self) -> None:
        vhdl_components = self.lstm.files(self.translation_args)

        target_components = [
            (RomFile, "wi_rom.vhd"),
            (RomFile, "wf_rom.vhd"),
            (RomFile, "wg_rom.vhd"),
            (RomFile, "wo_rom.vhd"),
            (RomFile, "bi_rom.vhd"),
            (RomFile, "bf_rom.vhd"),
            (RomFile, "bg_rom.vhd"),
            (RomFile, "bo_rom.vhd"),
            (SigmoidComponent, "sigmoid.vhd"),
            (TanhComponent, "tanh.vhd"),
            (LSTMFile, "lstm.vhd"),
            (LSTMCommonVHDLFile, "lstm_common.vhd"),
            (DualPort2ClockRamVHDLFile, "dual_port_2_clock_ram.vhd"),
        ]
        actual_components = [(type(x), x.name) for x in vhdl_components]

        self.assertEqual(target_components, actual_components)
