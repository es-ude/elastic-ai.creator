from unittest import TestCase

import torch

from elasticai.creator.nn.lstm import FixedPointLSTMWithHardActivations
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.pytorch.build_functions import (
    build_fixed_point_lstm,
)


def arange_parameter(
    start: int, end: int, shape: tuple[int, ...]
) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.reshape(torch.arange(start, end, dtype=torch.float32), shape)
    )


class LSTMBuildFunctionTest(TestCase):
    def test_build_lstm_layer_weights_correct_set(self) -> None:
        lstm = FixedPointLSTMWithHardActivations(
            input_size=1,
            hidden_size=1,
            bias=True,
            batch_first=True,
            fixed_point_factory=FixedPoint.get_factory(16, 8),
        )
        lstm.cell.linear_ih.weight = arange_parameter(start=0, end=4, shape=(4, 1))
        lstm.cell.linear_hh.weight = arange_parameter(start=4, end=8, shape=(4, 1))
        lstm.cell.linear_ih.bias = arange_parameter(start=8, end=12, shape=(4,))
        lstm.cell.linear_hh.bias = arange_parameter(start=12, end=16, shape=(4,))

        lstm_module = build_fixed_point_lstm(
            lstm,
            layer_id="lstm1",
            work_library_name="work",
        )

        self.assertEqual(lstm_module.weights_ih, [[[0.0], [1.0], [2.0], [3.0]]])
        self.assertEqual(lstm_module.weights_hh, [[[4.0], [5.0], [6.0], [7.0]]])

        self.assertEqual(lstm_module.biases_ih, [[8.0, 9.0, 10.0, 11.0]])
        self.assertEqual(lstm_module.biases_hh, [[12.0, 13.0, 14.0, 15.0]])
