import unittest
from typing import Any, Callable

from elasticai.creator.mlframework.typing import Module
from elasticai.creator.vhdl.evaluators.fixed_point_config_finder import (
    DataLoader,
    FixedPointConfigFinder,
)


class MyModel(Module):
    def __init__(self, fn: Callable[[float], float]) -> None:
        self.fn = fn

    def __call__(self, x: float, *args: Any, **kwargs: Any) -> float:
        return self.fn(x)


def create_data_loader(input_sample: list[float], label: list[float]) -> DataLoader:
    input_sample_batch = [input_sample]
    label_batch = [label]
    labelled_inputs = (input_sample_batch, label_batch)
    data_loader = [labelled_inputs]
    return data_loader


class FixedPointConfigurationFinderTest(unittest.TestCase):
    def test_integers_up_to_three_yield_3total_0frac_bits(self) -> None:
        input_sample = [1.0]
        label = input_sample
        evaluator = FixedPointConfigFinder(
            MyModel(lambda x: x), create_data_loader(input_sample, label)
        )
        result = evaluator.run()
        expected = dict(total_bits=2, frac_bits=0)
        self.assertEqual(expected, result)

    def test_halved_numbers_summed_to_1_5_yield_2total_1frac_bit(self) -> None:
        input_sample = [0.5]
        label = input_sample
        evaluator = FixedPointConfigFinder(
            MyModel(lambda x: x), create_data_loader(input_sample, label)
        )
        result = evaluator.run()
        expected = dict(total_bits=2, frac_bits=1)
        self.assertEqual(expected, result)
