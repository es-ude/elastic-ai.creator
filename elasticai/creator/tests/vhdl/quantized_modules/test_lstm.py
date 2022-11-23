import unittest

import torch

from elasticai.creator.vhdl.quantized_modules.hard_sigmoid import _HardSigmoidBase
from elasticai.creator.vhdl.quantized_modules.hard_tanh import _HardTanhBase
from elasticai.creator.vhdl.quantized_modules.linear import _LinearBase
from elasticai.creator.vhdl.quantized_modules.lstm import _LSTMCellBase


def linear_factory(in_features: int, out_features: int, bias: bool) -> _LinearBase:
    return _LinearBase(in_features=in_features, out_features=out_features, bias=bias)


class LSTMCellBaseTest(unittest.TestCase):
    def test_lstm_cell_base_shape_equals_reference_lstm_cell(self) -> None:
        lstm_cell = _LSTMCellBase(
            input_size=3,
            hidden_size=2,
            bias=True,
            linear_factory=linear_factory,
            sigmoid_factory=lambda: _HardSigmoidBase(),
            tanh_factory=lambda: _HardTanhBase(),
        )
        reference_lstm_cell = torch.nn.LSTMCell(input_size=3, hidden_size=2, bias=True)

        inputs = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        actual_h, actual_c = lstm_cell(inputs)
        expected_h, expected_c = reference_lstm_cell(inputs)

        self.assertEqual(tuple(expected_h.shape), tuple(actual_h.shape))
        self.assertEqual(tuple(expected_c.shape), tuple(actual_c.shape))

    def test_lstm_cell_base_quantized_forward_raises_error(self) -> None:
        with self.assertRaises(NotImplementedError):
            ...
