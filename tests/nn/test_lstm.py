from functools import partial
from typing import Any, Optional

import torch

from elasticai.creator.nn.arithmetics import FloatArithmetics
from elasticai.creator.nn.lstm import LSTM
from elasticai.creator.nn.lstm_cell import LSTMCell
from elasticai.creator.tests.tensor_test_case import TensorTestCase


def create_lstm(
    input_size: int, hidden_size: int, bias: bool, batch_first: bool
) -> tuple[torch.nn.Module, torch.nn.Module]:
    lstm_args: dict[str, Any] = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=bias,
        batch_first=batch_first,
    )
    lstm = LSTM(**lstm_args, lstm_cell_factory=torch.nn.LSTMCell)
    reference_lstm = torch.nn.LSTM(**lstm_args)

    lstm.cell.weight_ih = reference_lstm.weight_ih_l0
    lstm.cell.weight_hh = reference_lstm.weight_hh_l0
    if bias:
        lstm.cell.bias_ih = reference_lstm.bias_ih_l0
        lstm.cell.bias_hh = reference_lstm.bias_hh_l0

    return lstm, reference_lstm


class OutputsZeroLSTMCell(LSTMCell):
    def __init__(self, input_size: int, hidden_size: int, bias: bool) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            arithmetics=FloatArithmetics(),
            sigmoid_factory=torch.nn.Sigmoid,
            tanh_factory=torch.nn.Tanh,
        )

    def forward(
        self, x: torch.Tensor, state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = super().forward(x, state)
        return torch.zeros_like(h), torch.zeros_like(c)


def input_data(shape: tuple[int, ...]) -> torch.Tensor:
    num_samples = 1
    for dim in shape:
        num_samples *= dim
    return torch.linspace(-3, 3, num_samples).reshape(shape)


class LSTMTest(TensorTestCase):
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

        tensor_round = partial(torch.round, decimals=4)
        self.assertTensorEqual(
            tensor_round(target_outputs), tensor_round(actual_outputs)
        )
        self.assertTensorEqual(tensor_round(target_h), tensor_round(actual_h))
        self.assertTensorEqual(tensor_round(target_c), tensor_round(actual_c))

    def test_lstm_equals_pytorch_lstm_on_batched_inputs(self) -> None:
        lstm, reference_lstm = create_lstm(
            input_size=2, hidden_size=3, bias=True, batch_first=False
        )
        inputs = input_data((4, 8, 2))
        self.assertLSTMOutputsEqual(lstm, reference_lstm, inputs)

    def test_lstm_batch_first_equals_pytorch_lstm(self) -> None:
        lstm, reference_lstm = create_lstm(
            input_size=2, hidden_size=3, bias=True, batch_first=True
        )
        inputs = input_data((8, 4, 2))
        self.assertLSTMOutputsEqual(lstm, reference_lstm, inputs)

    def test_lstm_equals_pytorch_lstm_without_batches(self) -> None:
        lstm, reference_lstm = create_lstm(
            input_size=2, hidden_size=3, bias=True, batch_first=True
        )
        inputs = input_data((4, 2))
        self.assertLSTMOutputsEqual(lstm, reference_lstm, inputs)

    def test_lstm_explicit_pass_state(self) -> None:
        lstm, reference_lstm = create_lstm(
            input_size=2, hidden_size=3, bias=True, batch_first=True
        )
        inputs = input_data((8, 4, 2))
        state = (input_data((1, 8, 3)), input_data((1, 8, 3)))
        self.assertLSTMOutputsEqual(lstm, reference_lstm, inputs, state)

    def test_lstm_uses_given_lstm_cell(self) -> None:
        lstm = LSTM(
            input_size=2,
            hidden_size=3,
            bias=True,
            batch_first=True,
            lstm_cell_factory=OutputsZeroLSTMCell,
        )

        inputs = input_data((8, 4, 2))
        expected_output = torch.zeros((8, 4, 3))
        expected_h, expected_c = torch.zeros((1, 8, 3)), torch.zeros((1, 8, 3))
        actual_output, (actual_h, actual_c) = lstm(inputs)

        self.assertTensorEqual(expected_output, actual_output)
        self.assertTensorEqual(expected_h, actual_h)
        self.assertTensorEqual(expected_c, actual_c)
