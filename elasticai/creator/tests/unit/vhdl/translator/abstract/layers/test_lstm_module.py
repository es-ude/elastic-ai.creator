from unittest import TestCase

from elasticai.creator.vhdl.code_files.dual_port_2_clock_ram_component import (
    DualPort2ClockRamComponent,
)
from elasticai.creator.vhdl.code_files.fp_hard_sigmoid_component import (
    FPHardSigmoidComponent,
)
from elasticai.creator.vhdl.code_files.fp_hard_tanh_component import FPHardTanhComponent
from elasticai.creator.vhdl.code_files.lstm_component import LSTMComponent
from elasticai.creator.vhdl.code_files.rom_component import RomFile
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl_for_deprecation.translator.abstract.layers import LSTMModule


class LSTMModuleTest(TestCase):
    def setUp(self) -> None:
        self.lstm = LSTMModule(
            weights_ih=[[[1, 2], [3, 4], [5, 6], [7, 8]]],
            weights_hh=[[[1], [2], [3], [4]]],
            biases_ih=[[1, 2, 3, 4]],
            biases_hh=[[5, 6, 7, 8]],
            work_library_name="work",
            layer_id="0",
            fixed_point_factory=FixedPoint.get_builder(total_bits=8, frac_bits=2),
        )

    def test_correct_number_of_components(self) -> None:
        vhdl_components = list(self.lstm.files)
        self.assertEqual(len(vhdl_components), 12)

    def test_contains_all_needed_components(self) -> None:
        vhdl_components = self.lstm.files

        target_components = [
            (RomFile, "wi_rom_0.vhd"),
            (RomFile, "wf_rom_0.vhd"),
            (RomFile, "wg_rom_0.vhd"),
            (RomFile, "wo_rom_0.vhd"),
            (RomFile, "bi_rom_0.vhd"),
            (RomFile, "bf_rom_0.vhd"),
            (RomFile, "bg_rom_0.vhd"),
            (RomFile, "bo_rom_0.vhd"),
            (FPHardSigmoidComponent, "fp_hard_sigmoid_0.vhd"),
            (FPHardTanhComponent, "fp_hard_tanh_0.vhd"),
            (LSTMComponent, "lstm_0.vhd"),
            (DualPort2ClockRamComponent, "dual_port_2_clock_ram_0.vhd"),
        ]
        actual_components = [(type(x), x.name) for x in vhdl_components]

        self.assertEqual(target_components, actual_components)
