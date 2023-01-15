import unittest
from functools import partial
from typing import Optional

import torch

from elasticai.creator.nn.lstm import _LSTMBase
from elasticai.creator.nn.lstm_cell import FixedPointLSTMCell
from elasticai.creator.tests.unit.nn.utils import to_list
from elasticai.creator.vhdl.number_representations import FixedPoint


def create_full_res_lstm(
    input_size: int, hidden_size: int, bias: bool, batch_first: bool
) -> tuple[torch.nn.Module, torch.nn.Module]:
    lstm_args = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=bias,
        batch_first=batch_first,
    )
    lstm = _LSTMBase(**lstm_args, lstm_cell_factory=torch.nn.LSTMCell)
    reference_lstm = torch.nn.LSTM(**lstm_args)

    lstm.cell.weight_ih = reference_lstm.weight_ih_l0
    lstm.cell.weight_hh = reference_lstm.weight_hh_l0
    if bias:
        lstm.cell.bias_ih = reference_lstm.bias_ih_l0
        lstm.cell.bias_hh = reference_lstm.bias_hh_l0

    return lstm, reference_lstm


class CalledException(Exception):
    ...


def exploding_forward(
    x: torch.Tensor, state: Optional[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    raise CalledException()


class LSTMBaseTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def assertLSTMOutputsEqual(
        self,
        actual_lstm: torch.nn.Module,
        reference_lstm: torch.nn.Module,
        inputs: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        actual_outputs, (actual_h, actual_c) = actual_lstm(inputs, state)
        target_outputs, (target_h, target_c) = reference_lstm(inputs, state)

        self.assertEqual(to_list(target_outputs), to_list(actual_outputs))
        self.assertEqual(to_list(target_h), to_list(actual_h))
        self.assertEqual(to_list(target_c), to_list(actual_c))

    def test_full_res_lstm_equals_pytorch_lstm(self) -> None:
        lstm, reference_lstm = create_full_res_lstm(
            input_size=2, hidden_size=3, bias=True, batch_first=False
        )
        inputs = torch.randn((4, 8, 2))
        self.assertLSTMOutputsEqual(lstm, reference_lstm, inputs)

    def test_full_res_lstm_batch_first_equals_pytorch_lstm(self) -> None:
        lstm, reference_lstm = create_full_res_lstm(
            input_size=2, hidden_size=3, bias=True, batch_first=True
        )
        inputs = torch.randn((8, 4, 2))
        self.assertLSTMOutputsEqual(lstm, reference_lstm, inputs)

    def test_full_res_lstm_explicit_pass_state(self) -> None:
        lstm, reference_lstm = create_full_res_lstm(
            input_size=2, hidden_size=3, bias=True, batch_first=True
        )
        inputs = torch.randn((8, 4, 2))
        state = (torch.randn(1, 8, 3), torch.randn(1, 8, 3))
        self.assertLSTMOutputsEqual(lstm, reference_lstm, inputs, state)

    def test_lstm_cell_forward_function_called_for_normal_inference(self) -> None:
        lstm = _LSTMBase(
            input_size=2,
            hidden_size=3,
            bias=True,
            batch_first=True,
            lstm_cell_factory=partial(
                FixedPointLSTMCell,
                fixed_point_factory=FixedPoint.get_factory(total_bits=16, frac_bits=8),
            ),
        )
        lstm.cell.forward = exploding_forward
        inputs = torch.randn((4, 2))
        with self.assertRaises(CalledException):
            _ = lstm(inputs)

    def test_lstm_cell_quant_forward_function_called_for_quant_inference(self) -> None:
        lstm = _LSTMBase(
            input_size=2,
            hidden_size=3,
            bias=True,
            batch_first=True,
            lstm_cell_factory=partial(
                FixedPointLSTMCell,
                fixed_point_factory=FixedPoint.get_factory(total_bits=16, frac_bits=8),
            ),
        )
        lstm.cell.quantized_forward = exploding_forward
        inputs = torch.randn((4, 2))
        with self.assertRaises(CalledException):
            _ = lstm.quantized_forward(inputs)
