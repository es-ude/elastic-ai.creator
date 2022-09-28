import unittest

import torch
from torch.nn import Linear, Module
from torch.nn.parameter import Parameter

# from elasticai.creator.mlframework.typing import Module
from elasticai.creator.vhdl.evaluators.fixed_point_config_finder import (
    DataLoader,
    FixedPointConfigFinder,
)

# class MyModel(Module):
#     def __init__(self, fn: Callable[[float], float]) -> None:
#         self.fn = fn
#
#     def __call__(self, x: float, *args: Any, **kwargs: Any) -> float:
#         return self.fn(x)


class MyModel(Module):
    def __init__(self, weight: float, bias: float) -> None:
        super().__init__()
        self.linear = Linear(in_features=1, out_features=1)

        self.linear.weight = Parameter(torch.ones(1, 1) * weight)
        self.linear.bias = Parameter(torch.ones(1) * bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def create_data_loader(samples: list[float], labels: list[float]) -> DataLoader:
    return [(samples, labels)]


class FixedPointConfigurationFinderTest(unittest.TestCase):
    def test_fixed_point_representation_with_0frac_bits(self) -> None:
        model = MyModel(weight=2, bias=3)
        data = create_data_loader(samples=[1, 2, 3], labels=[5, 7, 9])
        evaluator = FixedPointConfigFinder(model=model, data=data, total_bits=16)
        result = evaluator.run()
        expected = dict(total_bits=16, frac_bits=0)
        self.assertEqual(expected, result)


# class FixedPointConfigurationFinderTest(unittest.TestCase):
#     def test_integers_with_0mse_yield_2total_0frac_bits(self) -> None:
#         data_loader = create_data_loader(samples=[0.0, 0.0], labels=[1.0, 1.0])
#         evaluator = FixedPointConfigFinder(
#             model=MyModel(lambda x: 1.0), data=data_loader, total_bits=16
#         )
#         result = evaluator.run()
#         expected = dict(total_bits=2, frac_bits=0)
#         self.assertEqual(expected, result)
#
#     def test_halved_numbers_yield_3total_1frac_bit(self) -> None:
#         data_loader = create_data_loader(samples=[0.0, 1.0], labels=[0.5, 1.5])
#         evaluator = FixedPointConfigFinder(
#             model=MyModel(lambda x: x + 0.5), data=data_loader, total_bits=16
#         )
#         result = evaluator.run()
#         expected = dict(total_bits=3, frac_bits=1)
#         self.assertEqual(expected, result)
#
