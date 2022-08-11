from functools import partial
from unittest import TestCase

from elasticai.creator.vhdl.components.dual_port_2_clock_ram import DualPort2ClockRam
from elasticai.creator.vhdl.components.lstm import LSTM as LSTMComponent
from elasticai.creator.vhdl.components.lstm_common import LSTMCommon
from elasticai.creator.vhdl.components.rom import Rom
from elasticai.creator.vhdl.components.sigmoid import Sigmoid
from elasticai.creator.vhdl.components.tanh import Tanh
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers.lstm import (
    LSTM,
    LSTMTranslationArguments,
)


class LSTMTest(TestCase):
    def setUp(self) -> None:
        self.lstm = LSTM(
            weights_ih=[[[1, 2], [3, 4], [5, 6], [7, 8]]],
            weights_hh=[[[1], [2], [3], [4]]],
            biases_ih=[[1, 2, 3, 4]],
            biases_hh=[[5, 6, 7, 8]],
        )

        self.translation_args = LSTMTranslationArguments(
            fixed_point_factory=partial(FixedPoint, total_bits=8, frac_bits=2),
            sigmoid_resolution=(-2.5, 2.5, 256),
            tanh_resolution=(-1, 1, 256),
        )

    def test_correct_number_of_components(self) -> None:
        vhdl_components = list(self.lstm.translate(self.translation_args))
        self.assertEqual(len(vhdl_components), 13)

    def test_contains_all_needed_components(self) -> None:
        vhdl_components = self.lstm.translate(self.translation_args)

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
            (LSTMComponent, "lstm.vhd"),
            (LSTMCommon, "lstm_common.vhd"),
            (DualPort2ClockRam, "dual_port_2_clock_ram.vhd"),
        ]
        actual_components = [(type(x), x.file_name) for x in vhdl_components]

        self.assertEqual(actual_components, target_components)
