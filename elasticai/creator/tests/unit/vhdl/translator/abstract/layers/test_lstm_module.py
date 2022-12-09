from unittest import TestCase

from elasticai.creator.vhdl.components import (
    DualPort2ClockRamComponent,
    LSTMCommonComponent,
    LSTMComponent,
    RomComponent,
    SigmoidComponent,
    TanhComponent,
)
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers import (
    LSTMModule,
    LSTMTranslationArgs,
)


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
        vhdl_components = list(self.lstm.components(self.translation_args))
        self.assertEqual(len(vhdl_components), 13)

    def test_contains_all_needed_components(self) -> None:
        vhdl_components = self.lstm.components(self.translation_args)

        target_components = [
            (RomComponent, "wi_rom.vhd"),
            (RomComponent, "wf_rom.vhd"),
            (RomComponent, "wg_rom.vhd"),
            (RomComponent, "wo_rom.vhd"),
            (RomComponent, "bi_rom.vhd"),
            (RomComponent, "bf_rom.vhd"),
            (RomComponent, "bg_rom.vhd"),
            (RomComponent, "bo_rom.vhd"),
            (SigmoidComponent, "sigmoid.vhd"),
            (TanhComponent, "tanh.vhd"),
            (LSTMComponent, "lstm.vhd"),
            (LSTMCommonComponent, "lstm_common.vhd"),
            (DualPort2ClockRamComponent, "dual_port_2_clock_ram.vhd"),
        ]
        actual_components = [(type(x), x.file_name) for x in vhdl_components]

        self.assertEqual(actual_components, target_components)
