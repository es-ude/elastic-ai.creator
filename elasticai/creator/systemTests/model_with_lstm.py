import torch
import unittest
import random
import numpy as np
from elasticai.creator.layers import QLSTMCell
from elasticai.creator.model_reporter import ModelReport


# example code from here: https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html?highlight=lstm%20cell#torch.nn.LSTMCell
class LSTMSystemTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ensure_reproducibility()

    @staticmethod
    def ensure_reproducibility():
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    def define_lstm_cell(self):
        input_size = 10
        hidden_size = 20
        state_quantizer = lambda x: x
        weight_quantizer = lambda x: x
        cell = QLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            state_quantizer=state_quantizer,
            weight_quantizer=weight_quantizer,
        )
        return cell

    def test_input(self) -> None:
        cell = self.define_lstm_cell()
        input = torch.randn(2, 3, 10)  # (time_steps, batch, input_size)
        hx = torch.randn(3, 20)  # (batch, hidden_size)
        cx = torch.randn(3, 20)
        output = []
        for i in range(input.size()[0]):
            hx, cx = cell(input[i], (hx, cx))
            output.append(hx)
            model_reporter = ModelReport(
                model=cell,
                data=[["example_0", "example_1"], [input, 0], [hx, 0]],
                is_binary=True,
            )
        output = torch.stack(output, dim=0)
        print(output)


if __name__ == "__main__":
    unittest.main()
