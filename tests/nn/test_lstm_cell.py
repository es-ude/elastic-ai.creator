from dataclasses import dataclass

import torch
from torch.nn.parameter import Parameter

from elasticai.creator.nn.arithmetics import Arithmetics, FloatArithmetics
from elasticai.creator.nn.lstm_cell import LSTMCell
from elasticai.creator.tests.tensor_test_case import TensorTestCase


def tensor(x: list) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def create_lstm_cell_and_reference(
    input_size: int,
    hidden_size: int,
    bias: bool,
    arithmetics: Arithmetics = FloatArithmetics(),
) -> tuple[LSTMCell, torch.nn.LSTMCell]:
    cell = LSTMCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=bias,
        sigmoid_factory=torch.nn.Sigmoid,
        tanh_factory=torch.nn.Tanh,
        arithmetics=arithmetics,
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


@dataclass
class FixedMulArithmetics(FloatArithmetics):
    values: list[float]

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return tensor(self.values)


@dataclass
class FixedAddArithmetics(FloatArithmetics):
    values: list[float]

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return tensor(self.values)


class ZeroActivation(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class LSTMCellTest(TensorTestCase):
    def test_lstm_cell_without_bias_equals_torch_lstm_cell(self) -> None:
        cell, reference_cell = create_lstm_cell_and_reference(
            input_size=3, hidden_size=2, bias=False
        )
        inputs = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        actual_h, actual_c = cell(inputs)
        expected_h, expected_c = reference_cell(inputs)

        self.assertTensorEqual(expected_h, actual_h)
        self.assertTensorEqual(expected_c, actual_c)

    def test_lstm_cell_equals_torch_lstm_cell(self) -> None:
        cell, reference_cell = create_lstm_cell_and_reference(
            input_size=3, hidden_size=2, bias=True
        )
        inputs = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        actual_h, actual_c = cell(inputs)
        expected_h, expected_c = reference_cell(inputs)

        self.assertTensorEqual(expected_h, actual_h)
        self.assertTensorEqual(expected_c, actual_c)

    def test_lstm_cell_without_batched_inputs_equals_pytorch_lstm_cell(self) -> None:
        cell, reference_cell = create_lstm_cell_and_reference(
            input_size=4, hidden_size=2, bias=True
        )
        inputs = tensor([1, 2, 3, 4])
        actual_h, actual_c = cell(inputs)
        expected_h, expected_c = reference_cell(inputs)

        self.assertTensorEqual(expected_h, actual_h)
        self.assertTensorEqual(expected_c, actual_c)

    def test_lstm_cell_uses_mul_from_arithmetics(self) -> None:
        cell = LSTMCell(
            input_size=1,
            hidden_size=2,
            bias=True,
            arithmetics=FixedMulArithmetics([0.0]),
            sigmoid_factory=torch.nn.Sigmoid,
            tanh_factory=torch.nn.Tanh,
        )

        inputs = tensor([[1], [2]])
        h, c = cell(inputs)

        self.assertTensorEqual(h, [0.0])
        self.assertTensorEqual(c, [0.0])

    def test_lstm_cell_uses_add_from_arithmetics(self) -> None:
        cell = LSTMCell(
            input_size=1,
            hidden_size=2,
            bias=False,
            arithmetics=FixedAddArithmetics([0.0]),
            sigmoid_factory=torch.nn.Sigmoid,
            tanh_factory=torch.nn.Tanh,
        )

        inputs = tensor([[1], [2]])
        h, c = cell(inputs)

        self.assertTensorEqual(h, [0.0])
        self.assertTensorEqual(c, [0.0])

    def test_lstm_cell_uses_sigmoid_factory(self) -> None:
        cell = LSTMCell(
            input_size=1,
            hidden_size=2,
            bias=False,
            arithmetics=FloatArithmetics(),
            sigmoid_factory=ZeroActivation,
            tanh_factory=torch.nn.Tanh,
        )

        inputs = tensor([[1]])
        h, c = cell(inputs)

        self.assertTensorEqual(h, [[0.0, 0.0]])
        self.assertTensorEqual(c, [[0.0, 0.0]])

    def test_lstm_cell_uses_tanh_factory(self) -> None:
        cell = LSTMCell(
            input_size=1,
            hidden_size=2,
            bias=False,
            arithmetics=FloatArithmetics(),
            sigmoid_factory=torch.nn.Sigmoid,
            tanh_factory=ZeroActivation,
        )

        inputs = tensor([[1]])
        h, c = cell(inputs)

        self.assertTensorEqual(h, [[0.0, 0.0]])
        self.assertTensorEqual(c, [[0.0, 0.0]])
