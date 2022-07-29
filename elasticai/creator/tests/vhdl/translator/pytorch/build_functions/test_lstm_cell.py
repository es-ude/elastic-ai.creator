from unittest import TestCase

import torch.nn

from elasticai.creator.vhdl.translator.pytorch.build_functions import build_lstm_cell


def arange_parameter(
    start: int, end: int, shape: tuple[int, ...]
) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.reshape(torch.arange(start, end, dtype=torch.float32), shape)
    )


class LSTMCellBuildFunctionTest(TestCase):
    def setUp(self) -> None:
        self.lstm_cell = torch.nn.LSTMCell(input_size=1, hidden_size=1)
        self.lstm_cell.weight_ih = arange_parameter(start=0, end=4, shape=(4, 1))
        self.lstm_cell.weight_hh = arange_parameter(start=4, end=8, shape=(4, 1))
        self.lstm_cell.bias_ih = arange_parameter(start=8, end=12, shape=(4,))
        self.lstm_cell.bias_hh = arange_parameter(start=12, end=16, shape=(4,))

    def test_build_lstm_layer_weights_correct_set(self) -> None:
        lstm_cell_translatable = build_lstm_cell(self.lstm_cell)

        self.assertEqual(lstm_cell_translatable.weights_ii, [[0.0]])
        self.assertEqual(lstm_cell_translatable.weights_if, [[1.0]])
        self.assertEqual(lstm_cell_translatable.weights_ig, [[2.0]])
        self.assertEqual(lstm_cell_translatable.weights_io, [[3.0]])

        self.assertEqual(lstm_cell_translatable.weights_hi, [[4.0]])
        self.assertEqual(lstm_cell_translatable.weights_hf, [[5.0]])
        self.assertEqual(lstm_cell_translatable.weights_hg, [[6.0]])
        self.assertEqual(lstm_cell_translatable.weights_ho, [[7.0]])

        self.assertEqual(lstm_cell_translatable.bias_ii, [8.0])
        self.assertEqual(lstm_cell_translatable.bias_if, [9.0])
        self.assertEqual(lstm_cell_translatable.bias_ig, [10.0])
        self.assertEqual(lstm_cell_translatable.bias_io, [11.0])

        self.assertEqual(lstm_cell_translatable.bias_hi, [12.0])
        self.assertEqual(lstm_cell_translatable.bias_hf, [13.0])
        self.assertEqual(lstm_cell_translatable.bias_hg, [14.0])
        self.assertEqual(lstm_cell_translatable.bias_ho, [15.0])
