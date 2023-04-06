import torch

from elasticai.creator.base_modules.linear import Linear
from elasticai.creator.nn.binary_arithmetics import BinaryArithmetics


class TestLinearLayer:
    def test_binary_linear_layer(self) -> None:
        x = [-0.4, 0.1]
        weight = [[0.1, -0.8], [-1.2, 4.0]]
        expected = [-1.0, 1.0]
        linear = Linear(
            arithmetics=BinaryArithmetics(),
            in_features=len(x),
            out_features=len(weight),
            bias=False,
        )
        with torch.no_grad():
            linear.weight = torch.nn.Parameter(torch.tensor(weight))

        actual = linear(torch.tensor(x)).tolist()
        assert actual == expected
