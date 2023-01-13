import unittest

import torch
from torch.nn.parameter import Parameter

from elasticai.creator.tests.vhdl.quantized_modules.utils import to_list
from elasticai.creator.vhdl.quantized_modules.lstm_cell import _LSTMCellBase


def to_tensor(x: list) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


class LSTMCellBaseTest(unittest.TestCase):
    @staticmethod
    def create_lstm_cell_base_and_reference(
        input_size: int, hidden_size: int, bias: bool
    ) -> tuple[_LSTMCellBase, torch.nn.LSTMCell]:
        cell = _LSTMCellBase(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            linear_factory=torch.nn.Linear,
            sigmoid_factory=torch.nn.Sigmoid,
            tanh_factory=torch.nn.Tanh,
        )
        reference_cell = torch.nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size, bias=bias
        )

        def ones_like(x: torch.Tensor) -> torch.Tensor:
            return Parameter(torch.ones_like(x))

        cell.linear_ih.weight = ones_like(cell.linear_ih.weight)
        cell.linear_hh.weight = ones_like(cell.linear_hh.weight)
        reference_cell.weight_ih = ones_like(reference_cell.weight_ih)
        reference_cell.weight_hh = ones_like(reference_cell.weight_hh)

        if bias:
            cell.linear_ih.bias = ones_like(cell.linear_ih.bias)
            cell.linear_hh.bias = ones_like(cell.linear_hh.bias)
            reference_cell.bias_ih = ones_like(reference_cell.bias_ih)
            reference_cell.bias_hh = ones_like(reference_cell.bias_hh)

        return cell, reference_cell

    def test_lstm_cell_base_without_bias_equals_torch_lstm_cell(self) -> None:
        cell, reference_cell = self.create_lstm_cell_base_and_reference(
            input_size=3, hidden_size=2, bias=False
        )
        inputs = to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        actual_h, actual_c = cell(inputs)
        expected_h, expected_c = reference_cell(inputs)

        self.assertEqual(to_list(expected_h), to_list(actual_h))
        self.assertEqual(to_list(expected_c), to_list(actual_c))

    def test_lstm_cell_base_equals_torch_lstm_cell(self) -> None:
        cell, reference_cell = self.create_lstm_cell_base_and_reference(
            input_size=3, hidden_size=2, bias=True
        )
        inputs = to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        actual_h, actual_c = cell(inputs)
        expected_h, expected_c = reference_cell(inputs)

        self.assertEqual(to_list(expected_h), to_list(actual_h))
        self.assertEqual(to_list(expected_c), to_list(actual_c))

    def test_lstm_cell_base_quantized_forward_raises_error(self) -> None:
        cell, _ = self.create_lstm_cell_base_and_reference(
            input_size=3, hidden_size=2, bias=True
        )
        with self.assertRaises(NotImplementedError):
            inputs = torch.tensor([[1, 2, 3]], dtype=torch.float32)
            _ = cell.quantized_forward(inputs)
