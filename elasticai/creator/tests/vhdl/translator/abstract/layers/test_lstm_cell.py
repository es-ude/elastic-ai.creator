from functools import partial
from unittest import TestCase

from elasticai.creator.vhdl.components.dual_port_2_clock_ram import DualPort2ClockRam
from elasticai.creator.vhdl.components.lstm_cell import LSTMCell as LSTMCellComponent
from elasticai.creator.vhdl.components.lstm_common import LSTMCommon
from elasticai.creator.vhdl.components.rom import Rom
from elasticai.creator.vhdl.components.sigmoid import Sigmoid
from elasticai.creator.vhdl.components.tanh import Tanh
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers.lstm_cell import (
    LSTMCell,
    LSTMCellTranslationArguments,
)


class LSTMCellTest(TestCase):
    def setUp(self) -> None:
        self.cell = LSTMCell(
            weights_ii=[[1, 2], [3, 4]],
            weights_hi=[[2, 3], [4, 5]],
            weights_if=[[3, 4], [5, 6]],
            weights_hf=[[4, 5], [6, 7]],
            weights_ig=[[5, 6], [7, 8]],
            weights_hg=[[6, 7], [8, 9]],
            weights_io=[[7, 8], [9, 0]],
            weights_ho=[[8, 9], [0, 1]],
            bias_ii=[1, 2],
            bias_hi=[2, 3],
            bias_if=[3, 4],
            bias_hf=[4, 5],
            bias_ig=[5, 6],
            bias_hg=[6, 7],
            bias_io=[7, 8],
            bias_ho=[8, 9],
        )

        self.translation_args = LSTMCellTranslationArguments(
            fixed_point_factory=partial(FixedPoint, total_bits=8, frac_bits=2),
            sigmoid_resolution=(-2.5, 2.5, 256),
            tanh_resolution=(-1, 1, 256),
        )

    def test_correct_number_of_components(self) -> None:
        vhdl_components = list(self.cell.translate(self.translation_args))
        self.assertEqual(len(vhdl_components), 13)

    def test_contains_all_needed_components(self) -> None:
        vhdl_components = self.cell.translate(self.translation_args)

        target_components = [
            (Rom, "wi_rom.vhd"),
            (Rom, "wf_rom.vhd"),
            (Rom, "wg_rom.vhd"),
            (Rom, "wo_rom.vhd"),
            (Rom, "bi_rom.vhd"),
            (Rom, "bf_rom.vhd"),
            (Rom, "bg_rom.vhd"),
            (Rom, "bo_rom.vhd"),
            (Sigmoid, "sigmoid.vhd"),
            (Tanh, "tanh.vhd"),
            (LSTMCellComponent, "lstm_cell.vhd"),
            (LSTMCommon, "lstm_common.vhd"),
            (DualPort2ClockRam, "dual_port_2_clock_ram.vhd"),
        ]
        actual_components = [(type(x), x.file_name) for x in vhdl_components]

        self.assertEqual(actual_components, target_components)
